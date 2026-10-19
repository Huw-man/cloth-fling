"""Offline readers and integrity checks; simulator and torch are never imported."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .schema import SCHEMA_VERSION, transition_schema
from .storage import (
    file_sha256,
    json_bytes,
    load_json,
    read_bundle,
    relative_path,
    reset_family_hash,
    sha256,
)


def read_transitions(path, *, valid_only=False) -> pa.Table:
    """Read committed parts only. Invalid rows remain unless explicitly filtered."""
    root = Path(path).resolve()
    manifest = load_json(root, "manifest.json")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema")
    tables = [
        pq.ParquetFile(relative_path(root, part["path"])).read() for part in manifest["parts"]
    ]
    for table in tables:
        if not table.schema.equals(transition_schema(), check_metadata=True):
            raise ValueError("transition schema mismatch")
    table = pa.concat_tables(tables) if tables else pa.Table.from_pylist([], transition_schema())
    return table.filter(table["execution_valid"]) if valid_only else table


load_transitions = read_transitions


def read_rgb(path, reference) -> np.ndarray:
    """Load learner-facing RGB independently of depth/state support or Genesis."""
    with Image.open(relative_path(Path(path).resolve(), reference)) as image:
        if image.mode != "RGB" or image.format != "PNG":
            raise ValueError("observation must be RGB PNG")
        return np.asarray(image).copy()


def read_support(path, reference):
    return read_bundle(Path(path).resolve(), reference)


def verify_dataset(path) -> dict:
    """Return valid/ok, errors and counts. Corrupt input returns a report, never success."""
    root = Path(path).resolve()
    errors = []
    report = {
        "valid": False,
        "ok": False,
        "errors": errors,
        "attempts": 0,
        "valid_attempts": 0,
        "invalid_attempts": 0,
        "episodes": 0,
    }

    def require(test, message):
        if not test:
            errors.append(message)

    try:
        manifest = load_json(root, "manifest.json")
        require(manifest["schema_version"] == SCHEMA_VERSION, "unsupported schema version")
        require(manifest["simulation_only"] is True, "simulation_only metadata missing")
        require(manifest["learned_input"] == "RGB only", "learned input boundary changed")
        require(
            manifest["in_flight"] is None, "unresolved in-flight action; reopen writer to recover"
        )
        files = manifest["files"]
        for reference, info in files.items():
            try:
                target = relative_path(root, reference)
                require(target.is_file(), f"missing file: {reference}")
                if target.is_file():
                    require(
                        target.stat().st_size == info["size_bytes"], f"size mismatch: {reference}"
                    )
                    require(
                        file_sha256(target) == info["sha256"], f"checksum mismatch: {reference}"
                    )
            except (ValueError, OSError) as exc:
                errors.append(f"file {reference}: {exc}")

        def ref_check(ref):
            if ref is None:
                return
            reference = ref["path"]
            relative_path(root, reference)
            require(reference in files, f"unregistered reference: {reference}")
            require(
                files.get(reference, {}).get("sha256") == ref["sha256"],
                f"reference hash mismatch: {reference}",
            )

        # All structured artifact references, including nested NPZ support, are registered.
        def refs(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    ref_check(value)
                for child in value.values():
                    refs(child)
            elif isinstance(value, list):
                for child in value:
                    refs(child)

        refs(manifest)
        for reference in files:
            if reference.startswith("support/") and reference.endswith(".json"):
                refs(load_json(root, reference))
                read_bundle(root, reference)  # Validate NPZ keys and prohibit pickle loading.
        rows = read_transitions(root).to_pylist()
        episodes = manifest["episodes"]
        report.update(
            attempts=len(rows),
            episodes=len(episodes),
            valid_attempts=sum(r["execution_valid"] for r in rows),
            invalid_attempts=sum(not r["execution_valid"] for r in rows),
        )
        require(len(rows) == len(manifest["parts"]), "each part must contain exactly one attempt")
        require(
            [r["transition_id"] for r in rows] == list(range(len(rows))),
            "transition IDs are not sequential",
        )
        require(
            [e["episode_id"] for e in episodes] == list(range(len(episodes))),
            "episode IDs are not sequential",
        )
        run_ids = [run["run_id"] for run in manifest["runs"]]
        require(len(set(run_ids)) == len(run_ids), "duplicate run ID")
        for index, run in enumerate(manifest["runs"]):
            require(run["status"] in ("closed", "interrupted"), "run is still open")
            require(
                run["parent_run_id"] == (run_ids[index - 1] if index else None),
                f"run ancestry mismatch: {run['run_id']}",
            )
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["episode_id"]].append(row)
            for field in transition_schema():
                if not field.nullable:
                    require(row[field.name] is not None, f"null required field: {field.name}")
        require(
            [r["episode_id"] for r in rows] == sorted(r["episode_id"] for r in rows),
            "transition episodes are out of sequence",
        )
        require(set(grouped).issubset(set(range(len(episodes)))), "row references unknown episode")
        split_groups = {}

        def final_state_check(ref, label, valid=False):
            if not ref:
                return
            state = read_bundle(root, ref["path"])
            if isinstance(state, dict):
                if "free" in state:
                    require(
                        np.asarray(state["free"], dtype=bool).all(),
                        f"{label}: final state has attachments (free mask)",
                    )
                for key in ("attached_ids", "attached_particle_ids", "attachments"):
                    if key in state:
                        attached = state[key]
                        require(
                            attached is not None and len(attached) == 0,
                            f"{label}: final state has attachments",
                        )
                if "attachment_count" in state:
                    require(state["attachment_count"] == 0, f"{label}: final state has attachments")
                if valid:
                    for key in ("positions", "velocities"):
                        require(
                            key in state and np.isfinite(np.asarray(state[key])).all(),
                            f"{label}: valid final state has missing/nonfinite {key}",
                        )
            elif valid:
                require(False, f"{label}: valid final state must be a mapping")

        for index, episode in enumerate(episodes):
            label = f"episode {index}"
            require(
                episode["parent_episode_id"] == (index - 1 if index else None),
                f"{label}: episode ancestry mismatch",
            )
            require(episode["run_id"] in run_ids, f"{label}: unknown run")
            if episode["run_id"] in run_ids:
                run = manifest["runs"][run_ids.index(episode["run_id"])]
                require(
                    episode["parent_run_id"] == run["parent_run_id"],
                    f"{label}: parent run mismatch",
                )
            require(episode["status"] == "closed", f"{label}: episode is unfinished")
            require(bool(episode["end_reason"]), f"{label}: missing terminal reason")
            require(
                episode["terminated"] or episode["truncated"],
                f"{label}: no terminal/truncation flag",
            )
            for key in (
                "reset_group_id",
                "reset_id",
                "reset_content_sha256",
                "reset_family_sha256",
                "reset_input_sha256",
            ):
                if episode[key] is None:
                    continue
                group = (key, episode[key])
                previous = split_groups.setdefault(group, episode["split"])
                require(previous == episode["split"], f"split leakage for {key}: {episode[key]}")
            reset = read_bundle(root, episode["reset"]["path"])
            require(
                sha256(json_bytes(reset["config"])) == episode["config_sha256"],
                f"{label}: config hash mismatch",
            )
            require(reset["seed"] == episode["seed"], f"{label}: reset seed mismatch")
            source = reset.get("reset_source")
            require(
                reset["cloth_id"] == episode["cloth_id"]
                and reset["reset_kind"] == episode["reset_kind"],
                f"{label}: reset generator/cloth identity mismatch",
            )
            if "metadata" in episode or "episode_metadata" in reset:
                require(
                    isinstance(episode.get("metadata"), dict)
                    and episode.get("metadata") == reset.get("episode_metadata"),
                    f"{label}: episode metadata mismatch",
                )
            family_hash = reset_family_hash(
                config_sha256=sha256(json_bytes(reset["config"])),
                cloth_id=reset["cloth_id"],
                seed=reset["seed"],
                reset_kind=reset["reset_kind"],
                reset_source=source,
            )
            require(
                family_hash == episode["reset_family_sha256"],
                f"{label}: reset family hash mismatch",
            )
            require(
                (sha256(json_bytes(source)) if source is not None else None)
                == episode["reset_input_sha256"],
                f"{label}: reset input hash mismatch",
            )
            content = {
                "config": reset["config"],
                "state": source if source is not None else reset["state"],
                "seed": reset["seed"] if source is None and reset["state"] is None else None,
            }
            require(
                sha256(json_bytes(content)) == episode["reset_content_sha256"],
                f"{label}: reset content hash mismatch",
            )
            require(
                episode["reset_sha256"] == episode["reset"]["sha256"],
                f"{label}: reset reference mismatch",
            )
            episode_rows = grouped[index]
            require(len(episode_rows) == episode["steps"], f"{label}: episode step count mismatch")
            require(
                [r["step_id"] for r in episode_rows] == list(range(len(episode_rows))),
                f"{label}: step IDs are not sequential",
            )
            before_ref = episode["initial_observation"]
            if before_ref:
                require(
                    manifest["observations"].get(f"{index}:{before_ref['observation_id']}")
                    == before_ref,
                    f"{label}: reset observation registration mismatch",
                )
            else:
                require(
                    bool(episode.get("reset_error")) and not episode_rows,
                    f"{label}: missing reset observation without reset failure",
                )
            for step, row in enumerate(episode_rows):
                step_label = f"{label} step {step}"
                require(row["schema_version"] == SCHEMA_VERSION, f"{step_label}: schema mismatch")
                for key in (
                    "run_id",
                    "parent_run_id",
                    "parent_episode_id",
                    "split",
                    "reset_group_id",
                    "reset_id",
                    "reset_sha256",
                    "config_id",
                    "config_sha256",
                    "cloth_id",
                    "seed",
                ):
                    require(row[key] == episode[key], f"{step_label}: {key} mismatch")
                require(before_ref is not None, f"{step_label}: unknown previous observation")
                if before_ref:
                    require(
                        row["before_observation_id"] == before_ref["observation_id"],
                        f"{step_label}: pre/next observation ID discontinuity",
                    )
                    for kind in ("rgb", "support"):
                        require(
                            row[f"before_{kind}_path"] == before_ref[kind]["path"]
                            and row[f"before_{kind}_sha256"] == before_ref[kind]["sha256"],
                            f"{step_label}: pre/next {kind} discontinuity",
                        )
                for key, value in row.items():
                    if isinstance(value, float):
                        require(math.isfinite(value), f"{step_label}: nonfinite {key}; use null")
                    if key.endswith("_path") and value is not None:
                        ref_check({"path": value, "sha256": row[key[:-5] + "_sha256"]})
                after_id = row["after_observation_id"]
                before_ref = (
                    manifest["observations"].get(f"{index}:{after_id}") if after_id else None
                )
                for prefix in ("before", "after"):
                    oid = row[f"{prefix}_observation_id"]
                    if oid is None:
                        require(
                            all(
                                row[f"{prefix}_{kind}_{field}"] is None
                                for kind in ("rgb", "support")
                                for field in ("path", "sha256")
                            ),
                            f"{step_label}: null observation has payload references",
                        )
                        continue
                    obs = manifest["observations"].get(f"{index}:{oid}")
                    require(obs is not None, f"{step_label}: missing observation registration")
                    if obs:
                        for kind in ("rgb", "support"):
                            require(
                                row[f"{prefix}_{kind}_path"] == obs[kind]["path"]
                                and row[f"{prefix}_{kind}_sha256"] == obs[kind]["sha256"],
                                f"{step_label}: observation reference mismatch",
                            )
                for first, last, gain in [
                    ("coverage_before", "coverage_after", "coverage_gain"),
                    ("area_before_m2", "area_after_m2", "area_gain_m2"),
                ]:
                    expected = (
                        row[last] - row[first]
                        if row[first] is not None and row[last] is not None
                        else None
                    )
                    if expected is not None and not math.isfinite(expected):
                        expected = None
                    require(
                        _close(row[gain], expected), f"{step_label}: {gain} arithmetic mismatch"
                    )
                if row["execution_valid"]:
                    required = (
                        "coverage_before",
                        "coverage_after",
                        "area_before_m2",
                        "area_after_m2",
                        "flat_area_m2",
                        "coverage_gain",
                        "reward",
                        "action_u0",
                        "action_v0",
                        "action_u1",
                        "action_v1",
                        "simulated_seconds",
                        "wall_seconds",
                    )
                    require(
                        all(row[key] is not None for key in required) and after_id is not None,
                        f"{step_label}: valid outcome has missing metrics/observation",
                    )
                    require(
                        _close(row["reward"], row["coverage_gain"]),
                        f"{step_label}: reward mismatch",
                    )
                    require(
                        row["simulated_seconds"] is not None
                        and row["simulated_seconds"] >= 0
                        and row["wall_seconds"] is not None
                        and row["wall_seconds"] >= 0,
                        f"{step_label}: negative or missing timing",
                    )
                else:
                    require(
                        row["reward"] is None, f"{step_label}: invalid outcome has trainable reward"
                    )
                    require(bool(row["reason"]), f"{step_label}: invalid outcome lacks reason")
                    require(
                        row["terminated"] or row["truncated"],
                        f"{step_label}: invalid execution did not end episode",
                    )
                flat = row["flat_area_m2"]
                if flat is not None:
                    require(flat > 0, f"{step_label}: flat area must be positive")
                    if flat > 0:
                        for cov, area in [
                            ("coverage_before", "area_before_m2"),
                            ("coverage_after", "area_after_m2"),
                        ]:
                            if row[cov] is not None and row[area] is not None:
                                require(
                                    _close(row[cov], row[area] / flat),
                                    f"{step_label}: {cov} arithmetic mismatch",
                                )
                if step < len(episode_rows) - 1:
                    require(
                        not row["terminated"] and not row["truncated"],
                        f"{step_label}: action follows terminal outcome",
                    )
                final_state_check(
                    {"path": row["final_state_path"]} if row["final_state_path"] else None,
                    step_label,
                    valid=row["execution_valid"],
                )
            require(
                before_ref == episode["last_observation"], f"{label}: last observation mismatch"
            )
            if episode_rows:
                last = episode_rows[-1]
                # An interrupt between actions closes episode metadata without rewriting immutable rows.
                if episode.get("closure_source") == "outcome":
                    require(
                        last["terminated"] == episode["terminated"]
                        and last["truncated"] == episode["truncated"],
                        f"{label}: final row terminal flags mismatch",
                    )
            final_state_check(
                episode.get("final_state"),
                label,
                valid=bool(episode_rows and episode_rows[-1]["execution_valid"]),
            )
            if episode.get("collection_summary"):
                summary = read_bundle(root, episode["collection_summary"]["path"])
                for key in (
                    "episode_id",
                    "seed",
                    "split",
                    "reset_group_id",
                    "cloth_id",
                    "reset_id",
                    "config_sha256",
                    "end_reason",
                    "metadata",
                ):
                    require(
                        summary[key] == episode[key], f"{label}: collection summary {key} mismatch"
                    )
                valid_count = sum(row["execution_valid"] for row in episode_rows)
                require(
                    summary["attempts"] == len(episode_rows)
                    and summary["valid"] == valid_count
                    and summary["invalid"] == len(episode_rows) - valid_count,
                    f"{label}: collection summary attempt counts mismatch",
                )
                expected_lifecycle = (
                    "initially_unfolded"
                    if episode["end_reason"] == "initially_unfolded"
                    else "unfolded_after_action"
                    if episode["end_reason"] == "settled_coverage_threshold"
                    else "stopped_without_threshold"
                )
                require(
                    summary["lifecycle"] == expected_lifecycle,
                    f"{label}: collection summary lifecycle mismatch",
                )
                initial_id = (
                    episode["initial_observation"]["observation_id"]
                    if episode["initial_observation"]
                    else None
                )
                require(
                    summary["initial_observation_id"] == initial_id,
                    f"{label}: collection summary initial observation mismatch",
                )
                initial_source = summary["initial_settlement_source"]
                confirmed = summary["initial_confirmed_settled_coverage"]
                confirmed_id = initial_id if initial_source else None
                confirmed_source = initial_source
                if initial_source:
                    velocities = np.asarray((reset.get("state") or {}).get("velocities", []))
                    velocity_valid = (
                        velocities.ndim == 2
                        and velocities.shape[1] == 3
                        and len(velocities) > 0
                        and np.isfinite(velocities).all()
                    )
                    rms = (
                        float(np.sqrt(np.mean(np.sum(velocities**2, axis=1))))
                        if velocity_valid
                        else None
                    )
                    require(
                        initial_id is not None
                        and confirmed is not None
                        and _close(confirmed, summary["initial_coverage"])
                        and rms is not None
                        and rms < 0.01
                        and _close(rms, summary["initial_rms_speed_m_s"]),
                        f"{label}: initial settled coverage lacks velocity/observation evidence",
                    )
                    require(
                        (source is None and initial_source == "successful_reset_contract")
                        or (
                            source is not None
                            and initial_source == "certified_restored_fixture"
                            and episode["metadata"].get("initial_settled") is True
                        ),
                        f"{label}: initial settlement lacks reset certification",
                    )
                    initial_state = reset.get("state") or {}
                    for key in ("attached_ids", "attached_particle_ids", "attachments"):
                        if key in initial_state:
                            require(
                                initial_state[key] is not None and len(initial_state[key]) == 0,
                                f"{label}: initial settled state has attachments",
                            )
                    if "free" in initial_state:
                        require(
                            np.asarray(initial_state["free"], dtype=bool).all(),
                            f"{label}: initial settled state has attachments",
                        )
                    if "attachment_count" in initial_state:
                        require(
                            initial_state["attachment_count"] == 0,
                            f"{label}: initial settled state has attachments",
                        )
                else:
                    require(confirmed is None, f"{label}: unconfirmed initial settled coverage")
                for row in episode_rows:
                    evidence = read_bundle(root, row["evidence_path"])
                    diagnostics = evidence["diagnostics"]
                    if (
                        row["execution_valid"]
                        and diagnostics.get("settled") is True
                        and diagnostics.get("terminal_observation_valid") is not False
                        and row["after_observation_id"] is not None
                        and row["after_observation_id"] != row["before_observation_id"]
                    ):
                        confirmed = row["coverage_after"]
                        confirmed_id = row["after_observation_id"]
                        confirmed_source = "valid_settled_action"
                require(
                    _close(summary["last_confirmed_settled_coverage"], confirmed)
                    and summary["last_confirmed_settled_observation_id"] == confirmed_id
                    and summary["last_confirmed_settlement_source"] == confirmed_source,
                    f"{label}: last confirmed settled coverage mismatch",
                )
                last_observed = (
                    episode_rows[-1]["coverage_after"]
                    if episode_rows
                    else summary["initial_coverage"]
                )
                last_id = (
                    episode["last_observation"]["observation_id"]
                    if episode["last_observation"]
                    else None
                )
                require(
                    _close(summary["last_observed_coverage"], last_observed)
                    and summary["last_observation_id"] == last_id
                    and summary["last_observed_execution_valid"]
                    == (episode_rows[-1]["execution_valid"] if episode_rows else None),
                    f"{label}: observed terminal coverage mismatch",
                )
                if episode["end_reason"] in ("initially_unfolded", "settled_coverage_threshold"):
                    require(
                        episode["terminated"] and not episode["truncated"],
                        f"{label}: threshold closure terminal flags mismatch",
                    )
                    threshold = summary["settled_coverage_threshold"]
                    require(
                        threshold is not None
                        and 0 < threshold <= 1
                        and confirmed is not None
                        and confirmed >= threshold,
                        f"{label}: threshold closure lacks confirmed settled coverage",
                    )
                    require(
                        (episode["end_reason"] == "initially_unfolded" and not episode_rows)
                        or (
                            episode["end_reason"] == "settled_coverage_threshold"
                            and bool(episode_rows)
                            and episode_rows[-1]["execution_valid"]
                            and confirmed_id == episode_rows[-1]["after_observation_id"]
                        ),
                        f"{label}: threshold closure lifecycle mismatch",
                    )
        for obs in manifest["observations"].values():
            rgb = read_rgb(root, obs["rgb"]["path"])
            support = read_bundle(root, obs["support"]["path"])
            require(
                support["observation_id"] == obs["observation_id"],
                "observation support ID mismatch",
            )
            depth = support["depth"]
            require(depth is None or depth.shape == rgb.shape[:2], "depth/RGB shape mismatch")
        committed = set(files) | {"manifest.json", ".writer.lock"}
        report["orphan_files"] = sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and str(p.relative_to(root)) not in committed
        )
    except Exception as exc:  # noqa: BLE001 - malformed input must produce a failed report.
        errors.append(f"{type(exc).__name__}: {exc}")
    report["valid"] = report["ok"] = not errors
    return report


def _close(value, expected):
    if value is None or expected is None:
        return value is expected
    return (
        math.isfinite(value)
        and math.isfinite(expected)
        and math.isclose(value, expected, rel_tol=1e-7, abs_tol=1e-9)
    )
