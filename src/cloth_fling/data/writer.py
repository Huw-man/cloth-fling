"""Single-writer local dataset with a manifest as its atomic commit point."""

from __future__ import annotations

import copy
import fcntl
import io
import math
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .schema import SCHEMA_VERSION, transition_schema
from .storage import (
    atomic_write,
    file_sha256,
    json_bytes,
    jsonable,
    load_json,
    relative_path,
    reset_family_hash,
    sha256,
    write_bundle,
)


class DatasetWriter:
    """See docs/dataset.md for durable intent, reset, and outcome APIs."""

    def __init__(self, output_dir, metadata=None):
        self.root = Path(output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = relative_path(self.root, ".writer.lock").open("a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("dataset already has an active writer") from None
        self.closed = False
        self.failed = False
        try:
            if (self.root / "manifest.json").exists():
                self.manifest = load_json(self.root, "manifest.json")
                if self.manifest["schema_version"] != SCHEMA_VERSION:
                    raise ValueError("unsupported dataset schema")
                # Do not extend a damaged dataset, including damaged intent support.
                for path, info in self.manifest["files"].items():
                    if file_sha256(relative_path(self.root, path)) != info["sha256"]:
                        raise ValueError(f"checksum mismatch: {path}")
            else:
                self.manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "simulation_only": True,
                    "learned_input": "RGB only",
                    "files": {},
                    "runs": [],
                    "episodes": [],
                    "parts": [],
                    "observations": {},
                    "in_flight": None,
                }
            self._recover()
            parent = self.manifest["runs"][-1]["run_id"] if self.manifest["runs"] else None
            self.run_id = uuid4().hex
            self.metadata = jsonable(metadata or {})
            self.manifest["runs"].append(
                {
                    "run_id": self.run_id,
                    "parent_run_id": parent,
                    "status": "open",
                    "started_unix": time.time(),
                    "metadata": self.metadata,
                }
            )
            if "DATASET_CARD.md" not in self.manifest["files"]:
                self._put("DATASET_CARD.md", DATASET_CARD.encode())
            self._commit()
        except BaseException:
            self._lock.close()
            raise

    def _commit(self):
        atomic_write(self.root, "manifest.json", json_bytes(self.manifest))

    @contextmanager
    def _transaction(self):
        if self.closed or self.failed:
            raise RuntimeError("writer is closed or failed; reopen to reconcile durable intent")
        previous = copy.deepcopy(self.manifest)
        try:
            yield
            self._commit()
        except BaseException:
            self.manifest = previous
            self.failed = True
            raise

    def _put(self, path, data):
        target = relative_path(self.root, path)
        if target.exists():
            raise FileExistsError(f"immutable artifact already exists: {path}")
        atomic_write(self.root, path, data)
        ref = {"path": path, "sha256": sha256(data)}
        self.manifest["files"][path] = {"sha256": ref["sha256"], "size_bytes": len(data)}
        return ref

    def _bundle(self, payload, category):
        return write_bundle(self._put, payload, f"support/{category}-{uuid4().hex}")

    def _episode(self, episode_id):
        return self.manifest["episodes"][episode_id]

    def _observation(self, episode_id, observation):
        rgb = np.asarray(observation.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB must be an H x W x 3 uint8 array")
        depth = None if observation.depth is None else np.asarray(observation.depth)
        if depth is not None and depth.shape != rgb.shape[:2]:
            raise ValueError("depth shape must match RGB height and width")
        fingerprint = sha256(
            json_bytes(
                {
                    "rgb": sha256(rgb.tobytes()),
                    "shape": rgb.shape,
                    "depth": None
                    if depth is None
                    else {"hash": sha256(depth.tobytes()), "dtype": str(depth.dtype)},
                    "calibration": observation.calibration,
                }
            )
        )
        key = f"{episode_id}:{observation.observation_id}"
        previous = self.manifest["observations"].get(key)
        if previous:
            if previous["fingerprint"] != fingerprint:
                raise ValueError("observation ID reused with different content")
            return previous
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        ref = self._put(f"rgb/{uuid4().hex}.png", buf.getvalue())
        support = self._bundle(
            {
                "observation_id": observation.observation_id,
                "depth": depth,
                "calibration": observation.calibration,
            },
            "observation",
        )
        result = {
            "observation_id": observation.observation_id,
            "rgb": ref,
            "support": support,
            "fingerprint": fingerprint,
        }
        self.manifest["observations"][key] = result
        return result

    def write_episode_reset(
        self,
        observation,
        *,
        seed,
        config,
        state=None,
        split=None,
        reset_group_id=None,
        cloth_id=None,
        reset_id=None,
        config_id=None,
        reset_source=None,
        reset_kind="drop",
        reset_error=None,
        reset_evidence=None,
    ):
        """Commit a new episode even if it will have zero actions; return integer ID."""
        with self._transaction():
            if self.manifest["in_flight"]:
                raise RuntimeError("cannot reset while an action is in flight")
            if any(e["status"] == "open" for e in self.manifest["episodes"]):
                raise RuntimeError("finish the current episode before resetting")
            config = jsonable(config)
            config_hash = sha256(json_bytes(config))
            cloth_id = cloth_id or self.metadata.get("cloth_id", "cloth-default")
            family_hash = reset_family_hash(
                config_sha256=config_hash,
                cloth_id=cloth_id,
                seed=seed,
                reset_kind=reset_kind,
                reset_source=reset_source,
            )
            content_hash = sha256(
                json_bytes(
                    {
                        "config": config,
                        "state": reset_source if reset_source is not None else state,
                        "seed": seed if reset_source is None and state is None else None,
                    }
                )
            )
            episode_id = len(self.manifest["episodes"])
            reset = self._bundle(
                {
                    "seed": seed,
                    "config": config,
                    "state": state,
                    "reset_source": reset_source,
                    "reset_kind": reset_kind,
                    "cloth_id": cloth_id,
                },
                "reset",
            )
            if observation is None and not reset_error:
                raise ValueError("missing reset observation requires an explicit reset_error")
            obs = self._observation(episode_id, observation) if observation is not None else None
            run = self.manifest["runs"][-1]
            reset_id = reset_id or f"reset-{content_hash}"
            episode = {
                "episode_id": episode_id,
                "parent_episode_id": episode_id - 1 if episode_id else None,
                "run_id": self.run_id,
                "parent_run_id": run["parent_run_id"],
                "seed": int(seed),
                "split": split or self.metadata.get("split", "train"),
                "reset_group_id": reset_group_id
                or self.metadata.get("reset_group_id", f"family-{family_hash}"),
                "cloth_id": cloth_id,
                "reset_id": reset_id,
                "reset_sha256": reset["sha256"],
                "reset": reset,
                "reset_content_sha256": content_hash,
                "reset_family_sha256": family_hash,
                "reset_input_sha256": sha256(json_bytes(reset_source))
                if reset_source is not None
                else None,
                "reset_kind": reset_kind,
                "config_id": config_id or f"config-{config_hash}",
                "config_sha256": config_hash,
                "initial_observation": obs,
                "last_observation": obs,
                "steps": 0,
                "status": "open",
                "terminated": False,
                "truncated": False,
                "end_reason": None,
                "final_state": None,
                "reset_error": reset_error,
            }
            if reset_error:
                episode.update(
                    status="closed",
                    terminated=reset_error != "reset_timeout",
                    truncated=reset_error == "reset_timeout",
                    end_reason=reset_error,
                    closure_source="metadata",
                )
                episode["reset_evidence"] = self._bundle(reset_evidence or {}, "reset-error")
                if state is not None:
                    episode["final_state"] = self._bundle(state, "state")
            self.manifest["episodes"].append(episode)
        return episode_id

    save_reset = write_episode_reset

    def write_reset_failure(
        self,
        *,
        seed,
        config,
        reason,
        state=None,
        reset_source=None,
        evidence=None,
        reset_kind="drop",
    ):
        """Persist failed preparation as a zero-step episode, never an action transition."""
        return self.write_episode_reset(
            None,
            seed=seed,
            config=config,
            state=state,
            reset_source=reset_source,
            reset_kind=reset_kind,
            reset_error=reason,
            reset_evidence=evidence,
        )

    def begin_attempt(self, episode_id, action, *, before=None):
        """Durably journal intent BEFORE calling the executor. Never replay an intent."""
        with self._transaction():
            episode = self._episode(episode_id)
            if episode["status"] != "open" or self.manifest["in_flight"]:
                raise RuntimeError("episode closed or another action is in flight")
            ref = episode["last_observation"]
            if ref is None:
                raise ValueError("cannot act without a known observation")
            if before is not None and self._observation(episode_id, before) != ref:
                raise ValueError("before observation is not the previous after observation")
            self.manifest["in_flight"] = {
                "episode_id": episode_id,
                "step_id": episode["steps"],
                "before": ref,
                "action": jsonable(action),
                "started_unix": time.time(),
            }

    def append_outcome(
        self,
        episode_id,
        outcome=None,
        *,
        final_state=None,
        after=None,
        reason=None,
        terminated=None,
        truncated=None,
        end_reason=None,
        failure_evidence=None,
    ):
        """Commit an ActionOutcome, or an explicit failure for the journaled attempt."""
        with self._transaction():
            episode = self._episode(episode_id)
            if episode["status"] != "open":
                raise RuntimeError("episode is closed")
            pending = self.manifest["in_flight"]
            if pending and pending["episode_id"] != episode_id:
                raise ValueError("pending action belongs to a different episode")
            if outcome is None and not pending:
                raise ValueError("a failure requires journaled intent")
            before = (
                self._observation(episode_id, outcome.before)
                if outcome is not None
                else pending["before"]
            )
            if before != episode["last_observation"]:
                raise ValueError("broken pre/next observation continuity")
            action = jsonable(outcome.action) if outcome is not None else pending["action"]
            if pending and action != pending["action"]:
                raise ValueError("outcome action differs from journaled intent")
            after = outcome.after if outcome is not None else after
            after_ref = self._observation(episode_id, after) if after is not None else None
            fields = [
                "coverage_before",
                "coverage_after",
                "area_before_m2",
                "area_after_m2",
                "flat_area_m2",
                "simulated_seconds",
                "wall_seconds",
            ]
            raw = {name: getattr(outcome, name, None) for name in fields}
            finite = {
                k: float(v) if v is not None and math.isfinite(v) else None for k, v in raw.items()
            }
            valid = bool(outcome is not None and outcome.execution_valid)
            reason = (
                reason
                if reason is not None
                else (outcome.reason if outcome else "execution_failed")
            )

            def pixel_number(value):
                if isinstance(value, dict) and set(value) == {"__nonfinite__"}:
                    return float(value["__nonfinite__"])
                return float(value)

            pixels = np.asarray([[pixel_number(v) for v in pair] for pair in action["pixels"]])
            if pixels.shape != (2, 2):
                raise ValueError("action pixels must be two (u, v) pairs")
            if valid and (
                any(v is None for v in finite.values())
                or after_ref is None
                or finite["flat_area_m2"] <= 0
                or not np.isfinite(pixels).all()
                or finite["wall_seconds"] < 0
                or finite["simulated_seconds"] < 0
            ):
                valid, reason = False, "nonfinite_or_missing_outcome: " + reason
            terminated = (
                (outcome.terminated if outcome is not None else True)
                if terminated is None
                else terminated
            )
            truncated = (
                (outcome.truncated if outcome is not None else False)
                if truncated is None
                else truncated
            )
            if not valid and not truncated:
                terminated = True
            if not valid:
                end_reason = (
                    end_reason
                    if end_reason in ("interrupted", "max_wall_seconds")
                    else "execution_invalid"
                )
            state_ref = self._bundle(final_state, "state") if final_state is not None else None
            evidence = self._bundle(
                {
                    "diagnostics": outcome.diagnostics if outcome else {},
                    "evidence": outcome.evidence if outcome else {},
                    "raw_metrics": raw,
                    "failure_reason": reason if not valid else None,
                    "cleanup": failure_evidence,
                },
                "evidence",
            )
            row = {name: None for name in transition_schema().names}
            for name in [
                "run_id",
                "parent_run_id",
                "episode_id",
                "parent_episode_id",
                "split",
                "reset_group_id",
                "cloth_id",
                "reset_id",
                "reset_sha256",
                "config_id",
                "config_sha256",
                "seed",
            ]:
                row[name] = episode[name]
            row.update(
                schema_version=SCHEMA_VERSION,
                transition_id=len(self.manifest["parts"]),
                step_id=episode["steps"],
                primitive=action["primitive"],
                policy=action["policy"],
                execution_valid=valid,
                terminated=bool(terminated),
                truncated=bool(truncated),
                reason=str(reason),
                end_reason=end_reason,
                **finite,
            )
            for prefix, ref in [("before", before), ("after", after_ref)]:
                if ref:
                    row[f"{prefix}_observation_id"] = ref["observation_id"]
                    for kind in ["rgb", "support"]:
                        row[f"{prefix}_{kind}_path"] = ref[kind]["path"]
                        row[f"{prefix}_{kind}_sha256"] = ref[kind]["sha256"]
            for prefix, ref in [("final_state", state_ref), ("evidence", evidence)]:
                if ref:
                    row[f"{prefix}_path"], row[f"{prefix}_sha256"] = ref["path"], ref["sha256"]
            for key, value in zip(
                ["action_u0", "action_v0", "action_u1", "action_v1"], pixels.flat
            ):
                row[key] = float(value) if math.isfinite(value) else None
            for target, first, last in [
                ("coverage_gain", "coverage_before", "coverage_after"),
                ("area_gain_m2", "area_before_m2", "area_after_m2"),
            ]:
                row[target] = (
                    finite[last] - finite[first]
                    if finite[first] is not None and finite[last] is not None
                    else None
                )
                if row[target] is not None and not math.isfinite(row[target]):
                    row[target] = None
                    valid = False
                    reason = "nonfinite_gain: " + reason
                    end_reason = "execution_invalid"
                    row.update(
                        execution_valid=False,
                        reason=reason,
                        end_reason="execution_invalid",
                        terminated=True,
                    )
                    terminated = True
            row["reward"] = row["coverage_gain"] if valid else None
            buf = io.BytesIO()
            pq.write_table(
                pa.Table.from_pylist([row], schema=transition_schema()), buf, compression="zstd"
            )
            part = self._put(f"transitions/part-{uuid4().hex}.parquet", buf.getvalue())
            self.manifest["parts"].append(part)
            self.manifest["in_flight"] = None
            episode.update(
                steps=episode["steps"] + 1, last_observation=after_ref, final_state=state_ref
            )
            if terminated or truncated:
                episode.update(
                    status="closed",
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    end_reason=end_reason or reason or "environment_terminated",
                    closure_source="outcome",
                )
        return row

    def finish_episode(
        self,
        episode_id,
        *,
        reason,
        terminated=False,
        truncated=False,
        final_state=None,
        evidence=None,
    ):
        with self._transaction():
            if not terminated and not truncated:
                raise ValueError("episode closure needs a terminated or truncated flag")
            if self.manifest["in_flight"]:
                raise RuntimeError("cannot finish an in-flight action")
            episode = self._episode(episode_id)
            if episode["status"] != "open":
                return
            episode.update(
                status="closed",
                end_reason=reason,
                terminated=bool(terminated),
                truncated=bool(truncated),
                closure_source="metadata",
            )
            if final_state is not None:
                episode["final_state"] = self._bundle(final_state, "state")
            if evidence is not None:
                episode["final_evidence"] = self._bundle(evidence, "episode-evidence")

    def _recover(self):
        pending = self.manifest["in_flight"]
        if pending:
            self.append_outcome(
                pending["episode_id"],
                reason="interrupted_in_flight",
                truncated=True,
                terminated=False,
                end_reason="interrupted",
            )
        changed = bool(pending)
        for episode in self.manifest["episodes"]:
            if episode["status"] == "open":
                episode.update(
                    status="closed",
                    truncated=True,
                    end_reason="interrupted",
                    closure_source="metadata",
                )
                changed = True
        for run in self.manifest["runs"]:
            if run["status"] == "open":
                run.update(status="interrupted", ended_unix=time.time())
                changed = True
        if changed:
            self._commit()

    def close(self, *, interrupted=False):
        if self.closed:
            return
        try:
            if not self.failed:
                with self._transaction():
                    run = self.manifest["runs"][-1]
                    run.update(
                        status="interrupted"
                        if interrupted or self.manifest["in_flight"]
                        else "closed",
                        ended_unix=time.time(),
                    )
        finally:
            self.closed = True
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(interrupted=kind is not None)


DATASET_CARD = """# Cloth-fling simulation dataset card

Status: local simulation-only pipeline artifact. Primitive behavior is not qualified by storage.
Schema: cloth_fling transition v1. Learned input: RGB PNG only.
Privileged depth, calibration, reset state, final state, diagnostics and evidence are supporting
execution/evaluation artifacts. Assisted attachments are not physical pinch grasps.

Complete before any training/release (outside this milestone):
- Owner, collection date, code revision, Genesis/runtime versions:
- Cloth/material/camera/reset/configuration distribution:
- Run/episode counts, valid and invalid attempts, reasons and signed gain distribution:
- Split assignment and reset ancestry/group isolation:
- Reset and primitive behavior qualification evidence and limitations:
- Known failure modes, selection bias, intended uses and excluded uses:
- License, privacy and redistribution review:

See manifest.json for run metadata, episode lineage, interrupted intent recovery and checksums.
Invalid rows have null reward and must be filtered explicitly. Negative valid gains are retained.
"""
