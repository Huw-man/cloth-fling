"""Serial benchmark orchestration; imports the simulator only for execution stages."""

from __future__ import annotations

import copy
import fcntl
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from cloth_fling.contracts import GraspAction, SceneConfig
from cloth_fling.data.storage import atomic_write, file_sha256, json_bytes
from cloth_fling.sim.appearance import AppearanceConfig
from cloth_fling.sim.primitives import PrimitiveConfig


def _write(root, name, value):
    atomic_write(Path(root), name, json_bytes(value))


@contextmanager
def _lease(root):
    """One process owns all benchmark simulator work on this host."""
    root.mkdir(parents=True, exist_ok=True)
    with Path("/tmp/cloth-fling-rtx5080.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("exclusive simulator lease is already held") from exc
        _write(root, "gpu-lease.json", {"owner": "milestone3-coordinator", "active": True})
        try:
            yield
        finally:
            _write(root, "gpu-lease.json", {"owner": "milestone3-coordinator", "active": False})


def _restore(env, state):
    from cloth_fling.diagnostics import _fingerprint

    obs = env.restore(copy.deepcopy(state))
    first = _fingerprint(env)
    again = env.restore(copy.deepcopy(state))
    if first != _fingerprint(env) or obs.observation_id != again.observation_id:
        raise RuntimeError("benchmark_state_restoration_mismatch")
    if not np.array_equal(obs.rgb, again.rgb) or not np.array_equal(obs.depth, again.depth):
        raise RuntimeError("benchmark_observation_restoration_mismatch")
    return again


def _unsafe(report):
    """Scientific failures are retained; broken infrastructure stops dependent work."""
    text = json.dumps(report, default=str).lower()
    return (
        any(
            word in text
            for word in (
                "restore_mismatch",
                "restore_readback",
                "restore mismatch",
                'recording_failed": true',
                "contact_accounting_capture_failed",
                "trace_callback_failed",
                "observation_chain",
                "corrupt",
            )
        )
        or report.get("final_restore_valid") is False
    )


def _attempt(env, root, ledger, fixture, state, pair, config, tag, *, primitive="laydown"):
    from cloth_fling.diagnostics import run_diagnostic

    category = "diagnostic_laydown" if primitive == "laydown" else "diagnostic_fling"
    destination = root / "diagnostics" / tag
    if (destination / "diagnostic.json").exists():
        entries = json.loads((root / "benchmark-budget.json").read_text())["entries"]
        receipt = next((e for e in entries if e["key"] == tag), None)
        if (
            receipt is None
            or receipt["status"] != "finished"
            or receipt["result"].get("sha256") != file_sha256(destination / "diagnostic.json")
        ):
            raise ValueError("unreconciled or corrupted diagnostic; do not replay")
        report = json.loads((destination / "diagnostic.json").read_text())
        if report["config"] != asdict(config) or report["fixed_pixels"] != pair["pixels"]:
            raise ValueError("diagnostic cache configuration/pair mismatch")
        return report
    token = ledger.reserve(
        category,
        wall_seconds=180,
        key=tag,
        metadata={"fixture_id": fixture["fixture_id"], "split": "development"},
    )
    started = time.monotonic()
    try:
        _restore(env, state)
        report = run_diagnostic(
            env,
            seed=fixture["seed"],
            output_dir=destination,
            primitive_config=config,
            action=GraspAction(tuple(map(tuple, pair["pixels"])), policy="reviewed_visible_rest"),
            primitives=(primitive,),
            record_video=True,
            appearance=AppearanceConfig(),
        )
    except BaseException as exc:
        ledger.abandon(token, result={"error": f"{type(exc).__name__}: {exc}"})
        raise
    ledger.finish(
        token,
        wall_seconds=time.monotonic() - started,
        result={
            "diagnostic_ref": str((destination / "diagnostic.json").relative_to(root)),
            "sha256": file_sha256(destination / "diagnostic.json"),
            "execution_valid": report["execution_valid"],
        },
    )
    if _unsafe(report):
        raise RuntimeError(f"benchmark infrastructure stop: {tag}")
    return report


def _laydown_pass(rows):
    folds = {11: False, 17: False}
    drops = set()
    for row in rows:
        if not row["execution_valid"]:
            continue
        if row["kind"] == "fold" and row["seed"] in folds:
            folds[row["seed"]] |= row["coverage_after"] >= 0.95
        if row["kind"] == "drop" and row["coverage_gain"] >= 0.10:
            drops.add(row["fixture_id"])
    return all(folds.values()) and len(drops) >= 4


class _EpisodeGuard:
    """Audit one boundary restore and fresh observations without modifying physics."""

    def __init__(self, env, state):
        self.env, self.state = env, state
        self.restores = self.resets = self.actions = 0
        self.last_after_id = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        self.resets += 1
        raise RuntimeError("benchmark episode forbids reset")

    def restore(self, state):
        if self.restores or self.actions:
            raise RuntimeError("benchmark restore inside evolving episode")
        from cloth_fling.diagnostics import _fingerprint

        # The independent saved state hash is checked before the collector begins.
        obs = self.env.restore(state)
        if self.env.snapshot()["state_hash"] != self.state["state_hash"]:
            raise RuntimeError("benchmark state restore readback mismatch")
        self.restores += 1
        self.initial_fingerprint = _fingerprint(self.env)
        return obs


def _collect_case(
    env,
    root,
    ledger,
    fixture,
    state,
    config,
    *,
    directory,
    max_actions,
    category,
    key,
    selector_name="geometry_guided",
    episode_token=None,
    qualification_sha256=None,
    record_video=False,
    reviewed_first=None,
    threshold=None,
):
    from cloth_fling.collection import collect_episodes
    from cloth_fling.data import read_transitions, verify_dataset
    from cloth_fling.selection import select_action
    from cloth_fling.sim.primitives import execute_action

    guarded = _EpisodeGuard(env, state)
    meter = {"attempts": 0, "selections": [], "action_wall_seconds": 0.0}
    overhead = {
        "token": ledger.reserve("simulator", wall_seconds=180, key=f"{key}:setup"),
        "started": time.monotonic(),
    }

    def selector(observation, scene_config, **kwargs):
        if guarded.actions and observation.observation_id != guarded.last_after_id:
            raise RuntimeError("benchmark observation_chain mismatch")
        meter["selections"].append(observation.observation_id)
        if reviewed_first is not None and not guarded.actions:
            return GraspAction(
                tuple(map(tuple, reviewed_first["pixels"])), policy="reviewed_visible_rest"
            )
        return select_action(observation, scene_config, **kwargs)

    def executor(current, action, *, config, trace_callback=None):
        ledger.finish(
            overhead["token"],
            wall_seconds=time.monotonic() - overhead["started"],
            result={"episode_overhead": True},
        )
        overhead["token"] = None
        metadata = {
            "fixture_id": fixture["fixture_id"],
            "split": fixture["split"],
            "selector": selector_name,
            "episode_token": episode_token,
            "state_content_sha256": fixture["state_content_sha256"],
            "qualification_sha256": qualification_sha256,
        }
        token = ledger.reserve(
            category, wall_seconds=150, key=f"{key}:action:{meter['attempts']}", metadata=metadata
        )
        tick = time.monotonic()
        meter["attempts"] += 1
        guarded.actions += 1
        try:
            outcome = execute_action(current, action, config=config, trace_callback=trace_callback)
        except BaseException as exc:
            ledger.abandon(token, result={"error": str(exc)})
            raise
        elapsed = time.monotonic() - tick
        meter["action_wall_seconds"] += elapsed
        ledger.finish(
            token,
            wall_seconds=elapsed,
            result={"valid": outcome.execution_valid, "reason": outcome.reason},
        )
        overhead["token"] = ledger.reserve(
            "simulator", wall_seconds=180, key=f"{key}:after:{meter['attempts']}"
        )
        overhead["started"] = time.monotonic()
        guarded.last_after_id = outcome.after.observation_id if outcome.after is not None else None
        if outcome.before.observation_id != meter["selections"][-1]:
            raise RuntimeError("benchmark observation_chain execution mismatch")
        return outcome

    benchmark = {
        "manifest_sha256": file_sha256(root / "benchmark-manifest.json"),
        "fixture_id": fixture["fixture_id"],
        "selector": selector_name,
        "qualification_sha256": qualification_sha256,
    }
    started = time.monotonic()
    report = collect_episodes(
        guarded,
        directory,
        episodes=1,
        max_actions=max_actions,
        seed=fixture["seed"],
        policy=selector_name,
        primitive="laydown",
        primitive_config=config,
        selector=selector,
        executor=executor,
        max_wall_seconds=max_actions * 125 + 30,
        reset_state=state,
        reset_kind="drop",
        record_video=record_video,
        appearance=AppearanceConfig(),
        split=fixture["split"],
        reset_group_id=fixture["reset_group_id"],
        cloth_id=fixture["cloth_id"],
        episode_metadata={"benchmark": benchmark, "initial_settled": True},
        settled_coverage_threshold=threshold,
    )
    report["benchmark_chain_audit"] = {
        "restores": guarded.restores,
        "resets": guarded.resets,
        "selections": meter["selections"],
        "actions": guarded.actions,
    }
    report["benchmark_wall_seconds"] = time.monotonic() - started
    verified = verify_dataset(directory)
    report["dataset_verification"] = verified
    rows = read_transitions(directory).to_pylist()
    for before, after in zip(rows, rows[1:]):
        if before["after_observation_id"] != after["before_observation_id"]:
            raise RuntimeError("benchmark observation_chain dataset mismatch")
    if not verified["valid"] or _unsafe(report) or guarded.restores != 1 or guarded.resets:
        _write(root, str((directory / "benchmark-case.json").relative_to(root)), report)
        raise RuntimeError("benchmark artifact/restore/recording regression")
    _write(root, str((directory / "benchmark-case.json").relative_to(root)), report)
    if overhead["token"] is not None:
        ledger.finish(
            overhead["token"],
            wall_seconds=time.monotonic() - overhead["started"],
            result={"episode_storage_and_verification": True},
        )
    return report, rows


def _finalize_qualification(env, root, ledger, manifest, laydown, pairs):
    from cloth_fling.benchmark import load_fixture, record_qualification
    from cloth_fling.selection import SelectionConfig

    # The fling profile is frozen before this separate diagnostic track.
    fling_config = PrimitiveConfig()
    dev = [f for f in manifest["fixtures"] if f["split"] == "development"]
    flings = []
    for fixture in dev[:6]:
        tag = f"{fixture['fixture_id']}-fling"
        _attempt(
            env,
            root,
            ledger,
            fixture,
            load_fixture(root, fixture["fixture_id"]),
            pairs["pairs"][fixture["fixture_id"]][0],
            fling_config,
            tag,
            primitive="fling",
        )
        flings.append(
            {
                "fixture_id": fixture["fixture_id"],
                "diagnostic_ref": {
                    "path": f"diagnostics/{tag}/diagnostic.json",
                    "sha256": file_sha256(root / "diagnostics" / tag / "diagnostic.json"),
                },
                "primitive_config": asdict(fling_config),
                "pixels": pairs["pairs"][fixture["fixture_id"]][0]["pixels"],
                "stage": "frozen_fling",
            }
        )
    config = PrimitiveConfig(**laydown["config"])
    continuity = []
    completed = 0
    drops = [f for f in dev if f["kind"] == "drop"]
    for index, fixture in enumerate(drops[:4]):
        directory = root / "continuity" / fixture["fixture_id"]
        summary = directory / "benchmark-case.json"
        if summary.exists():
            report = json.loads(summary.read_text())
            from cloth_fling.data import read_transitions, verify_dataset

            if not verify_dataset(directory)["valid"]:
                raise RuntimeError("corrupted continuity dataset")
            rows = read_transitions(directory).to_pylist()
        else:
            report, rows = _collect_case(
                env,
                root,
                ledger,
                fixture,
                load_fixture(root, fixture["fixture_id"]),
                config,
                directory=directory,
                max_actions=2,
                category="diagnostic_continuity",
                key=f"continuity:{index}",
                record_video=True,
                reviewed_first=pairs["pairs"][fixture["fixture_id"]][0],
            )
        valid = len(rows) == 2 and all(r["execution_valid"] for r in rows)
        lifecycle = bool(rows and rows[0]["execution_valid"] and rows[0]["coverage_after"] >= 0.95)
        completed += valid
        continuity.append(
            {
                "fixture_id": fixture["fixture_id"],
                "two_valid_actions": valid,
                "lifecycle_check": lifecycle,
                "diagnostic_ref": {
                    "path": str(summary.relative_to(root)),
                    "sha256": file_sha256(summary),
                },
            }
        )
        if completed >= 2:
            break
    record = {
        "manifest_sha256": file_sha256(root / "benchmark-manifest.json"),
        "benchmark_identity": manifest["identity"],
        "passed": completed >= 2,
        "primitive_config": {"laydown": laydown["config"], "fling": asdict(fling_config)},
        "selection_config": asdict(SelectionConfig()),
        "laydown_runs": [
            {
                "fixture_id": r["fixture_id"],
                "diagnostic_ref": {
                    "path": r["diagnostic_ref"],
                    "sha256": file_sha256(root / r["diagnostic_ref"]),
                },
                "primitive_config": r["config"],
                "pixels": pairs["pairs"][r["fixture_id"]][r["pair_index"]]["pixels"],
                "stage": "revision" if "revision" in r["diagnostic_ref"] else "initial",
            }
            for r in laydown["validation"]
        ],
        "fling_runs": flings,
        "continuity": continuity,
        "laydown_passed": True,
        "continuity_passed": completed >= 2,
        "implementation_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in ("sim/primitives.py", "selection.py", "collection.py")
        },
    }
    result = record_qualification(root, record)
    return {"stage_passed": result["passed"], "qualification": result}


def _review(env, root, manifest):
    """Freeze visibly feasible diagnostic pairs before observing action outcomes."""
    from cloth_fling.benchmark import load_fixture
    from cloth_fling.diagnostics import reviewed_pair_candidates

    path = root / "reviewed-pairs.json"
    if path.exists():
        return json.loads(path.read_text())
    reviews = {}
    for fixture in manifest["fixtures"]:
        if fixture["split"] != "development":
            continue
        state = load_fixture(root, fixture["fixture_id"])
        obs = _restore(env, state)
        options = reviewed_pair_candidates(env, max_pairs=64, candidate_budget=256)
        for option in options:
            option["predicted_ids"] = option["predicted_grasp_particle_ids"]
        if not options:
            raise RuntimeError("no reviewed visible pair on development fixture")
        # Second pair has distinct material endpoints, rather than a neighboring pixel.
        first = options[0]
        second = next(
            (
                p
                for p in options[1:]
                if len(set(p["predicted_ids"]) & set(first["predicted_ids"])) == 0
            ),
            first,
        )
        reviews[fixture["fixture_id"]] = [first, second]
        from PIL import Image, ImageDraw

        for pair_index, pair in enumerate((first, second)):
            image = Image.fromarray(obs.rgb).resize((768, 768))
            draw = ImageDraw.Draw(image)
            scale = 768 / obs.rgb.shape[0]
            for index, point in enumerate(pair["pixels"]):
                u, v = np.asarray(point) * scale
                draw.ellipse((u - 8, v - 8, u + 8, v + 8), fill=("red", "blue")[index])
            draw.text((8, 8), f"{fixture['fixture_id']} pair {pair_index}", fill="black")
            target = root / "pair-review" / (fixture["fixture_id"] + f"-{pair_index}.png")
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
    document = {
        "rule": "visible feasible boundary, rest-chord rank; pre-action fixed",
        "pairs": reviews,
        "reviewed": False,
    }
    _write(root, path.name, document)
    return document


def _qualify(
    env,
    root,
    ledger,
    manifest,
    *,
    revision_checks=False,
    revision_config=None,
    revision_reason=None,
):
    from cloth_fling.benchmark import load_fixture

    if (root / "reviewed-pairs.json").exists():
        pairs = _review(env, root, manifest)
    else:
        token = ledger.reserve("simulator", wall_seconds=300, key="diagnostic-pair-review")
        started = time.monotonic()
        try:
            pairs = _review(env, root, manifest)
        except BaseException as exc:
            ledger.abandon(token, result={"pair_review_error": str(exc)})
            raise
        ledger.finish(
            token, wall_seconds=time.monotonic() - started, result={"pairs_frozen_for_review": True}
        )
    if not pairs.get("reviewed"):
        return {
            "stage_passed": False,
            "status": "awaiting_pair_review",
            "review": str(root / "reviewed-pairs.json"),
        }
    fixtures = [f for f in manifest["fixtures"] if f["split"] == "development"]
    candidate = replace(PrimitiveConfig(), laydown_distance_m=0.30)
    plan_path = root / "motion-plan.json"
    plan = {
        "reference": asdict(PrimitiveConfig()),
        "candidate": asdict(candidate),
        "fling": asdict(PrimitiveConfig()),
        "selection": "existing defaults",
        "pairs_sha256": file_sha256(root / "reviewed-pairs.json"),
        "primitive_implementation_sha256": file_sha256(
            Path(__file__).parent / "sim" / "primitives.py"
        ),
    }
    if plan_path.exists():
        original_plan = json.loads(plan_path.read_text())
        if not revision_checks and original_plan != plan:
            raise ValueError("frozen initial motion configuration mismatch")
        if original_plan["pairs_sha256"] != plan["pairs_sha256"]:
            raise ValueError("frozen reviewed pairs changed")
    else:
        if revision_checks:
            raise ValueError("revision requires measured initial diagnostics")
        _write(root, plan_path.name, plan)
    all_rows = []

    def diagnostic(f, config, pair_index, tag):
        report = _attempt(
            env,
            root,
            ledger,
            f,
            load_fixture(root, f["fixture_id"]),
            pairs["pairs"][f["fixture_id"]][pair_index],
            config,
            tag,
        )
        row = {
            k: report["runs"][0][k]
            for k in (
                "execution_valid",
                "reason",
                "coverage_before",
                "coverage_after",
                "coverage_gain",
            )
        }
        row.update(
            fixture_id=f["fixture_id"],
            seed=f["seed"],
            kind=f["kind"],
            diagnostic_ref=f"diagnostics/{tag}/diagnostic.json",
            config=asdict(config),
            pair_index=pair_index,
        )
        all_rows.append(row)
        _write(root, "laydown-progress.json", {"runs": all_rows})
        return row

    if revision_checks:
        all_rows = json.loads((root / "laydown-progress.json").read_text())["runs"]
        all_rows = [r for r in all_rows if "revision" not in r["diagnostic_ref"]]
        entries = json.loads((root / "benchmark-budget.json").read_text())["entries"]
        for row in all_rows:
            receipt = next(
                (
                    e
                    for e in entries
                    if e.get("result", {}).get("diagnostic_ref") == row["diagnostic_ref"]
                ),
                None,
            )
            if receipt is None or receipt["result"]["sha256"] != file_sha256(
                root / row["diagnostic_ref"]
            ):
                raise ValueError("initial diagnostic artifact changed before revision")
    else:
        for f in fixtures:
            if f["kind"] == "fold":
                diagnostic(f, PrimitiveConfig(), 0, f"{f['fixture_id']}-reference")
                diagnostic(f, candidate, 0, f"{f['fixture_id']}-candidate")
        for f in fixtures:
            if f["kind"] == "drop":
                row = diagnostic(f, candidate, 0, f"{f['fixture_id']}-candidate-0")
                if not (row["execution_valid"] and row["coverage_gain"] >= 0.10):
                    diagnostic(f, candidate, 1, f"{f['fixture_id']}-candidate-1")
    validation = [r for r in all_rows if r["config"] == asdict(candidate)]
    if not _laydown_pass(validation):
        if not revision_checks:
            return {
                "stage_passed": False,
                "status": "measured_revision_available",
                "runs": all_rows,
                "remaining_revision_checks": 8,
            }
        if revision_config is None or not revision_reason:
            raise ValueError("revision requires configuration and measured failure reason")
        revised = PrimitiveConfig(**json.loads(Path(revision_config).read_text()))
        original = PrimitiveConfig(**json.loads(plan_path.read_text())["candidate"])
        changes = {k for k, v in asdict(revised).items() if asdict(original)[k] != v}
        if any(not k.startswith("laydown_") for k in changes):
            raise ValueError("revision may change only targeted laydown trajectory settings")
        frozen = {
            "config": asdict(revised),
            "measured_reason": revision_reason,
            "pairs_sha256": file_sha256(root / "reviewed-pairs.json"),
            "primitive_implementation_sha256": plan["primitive_implementation_sha256"],
        }
        target = root / "revision-plan.json"
        if target.exists() and json.loads(target.read_text()) != frozen:
            raise ValueError("only one frozen trajectory revision is permitted")
        _write(root, target.name, frozen)
        validation = []
        for f in fixtures:
            if f["kind"] == "drop" or f["seed"] in (11, 17):
                validation.append(diagnostic(f, revised, 0, f"{f['fixture_id']}-revision"))
        candidate = revised
    passed = _laydown_pass(validation)
    summary = {
        "stage_passed": passed,
        "laydown_passed": passed,
        "status": "laydown_passed" if passed else "laydown_failed_stop",
        "config": asdict(candidate),
        "runs": all_rows,
        "validation": validation,
    }
    _write(root, "laydown-result.json", summary)
    return summary


def run_benchmark_stage(
    stage,
    output_dir,
    *,
    backend="gpu",
    revision_checks=False,
    revision_config=None,
    revision_reason=None,
):
    from cloth_fling.benchmark import BudgetLedger, prepare_benchmark

    root = Path(output_dir)
    if stage == "report":
        from cloth_fling.benchmark import report_benchmark

        return report_benchmark(root)
    if stage not in {"prepare", "qualify", "pilot"}:
        raise ValueError(f"unknown benchmark stage: {stage}")
    if stage != "prepare":
        from cloth_fling.benchmark import load_benchmark

        frozen = load_benchmark(root)
        if not frozen["stage_passed"]:
            return {
                "stage_passed": False,
                "status": "reset_generation_stop",
                "stop_reason": frozen["stop_reason"],
            }
        if stage == "pilot":
            from cloth_fling.benchmark import load_qualification

            if not load_qualification(root)["passed"]:
                raise ValueError("pilot requires passed qualification before scene construction")
    with _lease(root):
        from cloth_fling.sim.backend import GenesisBackend

        ledger = BudgetLedger(root)
        token = ledger.reserve("simulator", wall_seconds=180, key=f"build-{time.time_ns()}")
        started = time.monotonic()
        try:
            env = GenesisBackend(
                SceneConfig(contact_profile="native_coulomb_v2"),
                backend=backend,
                headless=True,
                appearance=AppearanceConfig(),
            )
        except BaseException as exc:
            ledger.abandon(token, result={"build_error": str(exc)})
            raise
        ledger.finish(token, wall_seconds=time.monotonic() - started, result={"built": True})
        try:
            if stage == "prepare":
                return prepare_benchmark(env, root, appearance=AppearanceConfig())
            from cloth_fling.benchmark import benchmark_identity, load_benchmark

            manifest = load_benchmark(
                root, identity=benchmark_identity(env, appearance=AppearanceConfig())
            )
            if manifest["stop_reason"] != "completed":
                return {
                    "stage_passed": False,
                    "status": "reset_generation_stop",
                    "stop_reason": manifest["stop_reason"],
                }
            if stage == "qualify":
                if (root / "benchmark-qualification.json").exists():
                    from cloth_fling.benchmark import load_qualification

                    qualification = load_qualification(root)
                    return {"stage_passed": qualification["passed"], "qualification": qualification}
                result = _qualify(
                    env,
                    root,
                    ledger,
                    manifest,
                    revision_checks=revision_checks,
                    revision_config=revision_config,
                    revision_reason=revision_reason,
                )
                if result.get("laydown_passed"):
                    return _finalize_qualification(
                        env,
                        root,
                        ledger,
                        manifest,
                        result,
                        json.loads((root / "reviewed-pairs.json").read_text()),
                    )
                return result
            if stage == "pilot":
                return _pilot(env, root, ledger, manifest)
            raise ValueError(f"unknown benchmark stage: {stage}")
        finally:
            env.close()


def _pilot(env, root, ledger, manifest):
    from cloth_fling.benchmark import load_fixture, load_qualification, report_benchmark

    qualification = load_qualification(root)
    if not qualification["passed"]:
        raise ValueError("pilot requires passed matching laydown and continuity qualification")
    for name, expected in qualification["implementation_sha256"].items():
        if file_sha256(Path(__file__).parent / name) != expected:
            raise ValueError("qualified motion/selection/collection implementation changed")
    qualification_hash = file_sha256(root / "benchmark-qualification.json")
    config = PrimitiveConfig(**qualification["primitive_config"]["laydown"])
    results = []
    drops = [f for f in manifest["fixtures"] if f["kind"] == "drop"]
    drops.sort(key=lambda f: (f["split"] == "heldout", f["seed"]))
    first = {
        split: next(f["fixture_id"] for f in drops if f["split"] == split)
        for split in ("development", "heldout")
    }
    prior = ledger.status()["pilot_episodes"]
    for fixture in drops:
        for selector in ("uniform_valid", "geometry_guided"):
            previous = next(
                (
                    e
                    for e in prior
                    if e["fixture_id"] == fixture["fixture_id"] and e["selector"] == selector
                ),
                None,
            )
            if previous is not None:
                if previous["status"] != "finished":
                    raise RuntimeError("unfinished pilot episode: preserve evidence; do not replay")
                results.append(previous["result"])
                continue
            episode_token = ledger.claim_pilot(
                fixture["fixture_id"], selector, qualification_sha256=qualification_hash
            )
            directory = root / "pilot" / fixture["split"] / fixture["fixture_id"] / selector
            report, _ = _collect_case(
                env,
                root,
                ledger,
                fixture,
                load_fixture(root, fixture["fixture_id"]),
                config,
                directory=directory,
                max_actions=5,
                category="pilot",
                key=f"pilot:{fixture['fixture_id']}:{selector}",
                selector_name=selector,
                episode_token=episode_token,
                qualification_sha256=qualification_hash,
                record_video=fixture["fixture_id"] == first[fixture["split"]],
                threshold=0.95,
            )
            result = {
                "fixture_id": fixture["fixture_id"],
                "selector": selector,
                "dataset": str(directory.relative_to(root)),
                "attempts": report["attempts"],
                "report_ref": {
                    "path": str((directory / "benchmark-case.json").relative_to(root)),
                    "sha256": file_sha256(directory / "benchmark-case.json"),
                },
            }
            ledger.finish_pilot(episode_token, result=result)
            results.append(result)
    report = report_benchmark(root)
    report["stage_passed"] = report.get("heldout_gate", {}).get("passed", False)
    return report
