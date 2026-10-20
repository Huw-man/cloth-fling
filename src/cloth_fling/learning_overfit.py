"""Bounded RGB single-reset overfitting; no physical resume or campaign extension."""

import argparse
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from cloth_fling import learning_pilot as pilot
from cloth_fling.benchmark_run import _reservation_deadline
from cloth_fling.contracts import scene_config_from_dict
from cloth_fling.data import read_support, read_transitions, verify_dataset
from cloth_fling.data.storage import file_sha256, relative_path
from cloth_fling.frozen_laydown import CONFIG_SHA256, frozen_laydown_spec
from cloth_fling.provenance import runtime_provenance
from cloth_fling.reference_pilot import certify_release
from cloth_fling.reference_run import reference_lease
from cloth_fling.reference_tune import read_hashed_bundle

DEFAULT_OUTPUT = "runs/single-reset-overfit-v1"
SOURCE_ROOT = Path("runs/image-learning-pilot-v1")
FIXTURE_ID = "fold-50-+00"
FIXTURE_SHA256 = "13a3c50ecc61382dbfb88e2b05ce758df5adfa0d6878c2ec49df9230a8eb2286"
STATE_HASH = "d1ec11b175fa15a62337f06ff08a276ba905afffefd65c3c94ab5694f915299a"
WALL_SECONDS = 6 * 3600
FINAL_RESERVE_SECONDS = 3600
ACTION_LIMIT = 300
PROTOCOL = {
    "version": "single_reset_overfit_v1",
    "fixture_id": FIXTURE_ID,
    "fixture_sha256": FIXTURE_SHA256,
    "state_hash": STATE_HASH,
    "frozen_config_sha256": CONFIG_SHA256,
    "coverage_stop": 0.98,
    "require_containment": True,
    "actions_per_episode": 3,
    "exploration_episodes": 20,
    "refresh_rounds": 7,
    "refresh_episodes_per_round": 10,
    "refresh_epsilon": 0.5,
    "confirmation_episodes": 2,
    "training_steps": 2000,
    "seed": 1001,
    "wall_seconds": WALL_SECONDS,
    "final_reserve_seconds": FINAL_RESERVE_SECONDS,
    "action_limit": ACTION_LIMIT,
    "old_labels_or_weights": False,
}
write, read, claim = pilot.write, pilot.read, pilot.claim


def schedule():
    result = [("exploration", "collection", 20), ("fit-00", "fit", 0), ("check-00", "check", 1)]
    for index in range(1, 8):
        result.extend(
            [
                (f"refresh-{index:02d}", "collection", 10),
                (f"fit-{index:02d}", "fit", 0),
                (f"check-{index:02d}", "check", 1),
            ]
        )
    return result


class Journal:
    """New-root-only journal. Intent files remain authoritative after partial writes."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.stage, self.kind = "prepare", "prepare"
        self.document = {
            "status": "running_never_resume",
            "outcome": "incomplete",
            "started_unix": time.time(),
            "charged_seconds": 0.0,
            "stages": [],
            "actions": [],
            "episodes": [],
        }
        self.save()
        write(self.root, "protocol.json", PROTOCOL)
        write(self.root, "implementation.json", pilot.implementation_identity())
        write(self.root, "runtime.json", runtime_provenance())
        write(self.root, "frozen-laydown.json", frozen_laydown_spec())

    def remaining(self):
        return WALL_SECONDS - (time.monotonic() - self.started)

    def save(self):
        self.document["charged_seconds"] = time.monotonic() - self.started
        write(self.root, "ledger.json", self.document)

    @property
    def reserve(self):
        return 0 if self.kind in ("confirmation", "replays", "report") else FINAL_RESERVE_SECONDS

    def check(self, *, reserve=None):
        if self.remaining() <= (self.reserve if reserve is None else reserve):
            raise TimeoutError("single-reset wall limit or final reserve reached")

    def claim_action(self, episode, step):
        self.check(reserve=self.reserve + 180)
        if step not in range(3) or len(self.document["actions"]) >= ACTION_LIMIT:
            raise RuntimeError("single-reset action budget exhausted")
        if any(x["episode"] == episode and x["step"] == step for x in self.document["actions"]):
            raise RuntimeError("action already claimed; retry forbidden")
        intent = {
            "index": len(self.document["actions"]),
            "stage": self.stage,
            "episode": episode,
            "step": step,
            "status": "claimed_never_retry",
        }
        claim(self.root / "action-intents" / f"{intent['index']:04d}.json", intent)
        self.document["actions"].append(intent)
        self.save()

    @contextmanager
    def operation(self, stage, kind):
        self.stage, self.kind = stage, kind
        self.check()
        claim(self.root / "stage-intents" / f"{stage}.json", {"stage": stage, "kind": kind})
        entry = {"stage": stage, "kind": kind, "status": "claimed_never_retry", "outputs": {}}
        self.document["stages"].append(entry)
        self.save()
        print(
            f"stage {stage}: starting, {len(self.document['actions'])}/{ACTION_LIMIT} actions claimed",
            flush=True,
        )
        try:
            yield entry["outputs"]
            self.check()
            entry["status"] = "complete"
        except BaseException as exc:
            entry.update(status="failed_or_uncertain", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.save()

    def seal(self, outputs, path):
        path = Path(path)
        outputs[str(path.relative_to(self.root))] = file_sha256(path)


def prepare_fixture(journal, source_root=SOURCE_ROOT):
    """Copy the archived bundle bytes and verify their frozen native identity."""
    source_root = Path(source_root)
    records = read(source_root, "fixtures-manifest.json")["cases"]
    case = next(c for c in records if c["id"] == FIXTURE_ID)
    if case["state_hash"] != STATE_HASH or case["fixture"]["sha256"] != FIXTURE_SHA256:
        raise ValueError("archived fixture identity changed")
    source = relative_path(source_root, case["fixture"]["path"])
    if file_sha256(source) != FIXTURE_SHA256:
        raise ValueError("archived fixture bytes changed")
    native = read_hashed_bundle(source.parent, source.name)
    if native["state_hash"] != STATE_HASH or not native["free"].all():
        raise ValueError("archived native fixture changed or attached")
    target = journal.root / "fixture"
    target.mkdir()
    payload = json.loads(source.read_text())
    arrays = relative_path(source.parent, payload["arrays"]["path"])
    shutil.copy2(arrays, target / arrays.name)
    shutil.copy2(source, target / source.name)
    copied = read_hashed_bundle(target, source.name)
    if copied["state_hash"] != STATE_HASH or file_sha256(target / source.name) != FIXTURE_SHA256:
        raise ValueError("fixture copy changed identity")
    case = dict(case, fixture={"path": "fixture/fixture.json", "sha256": FIXTURE_SHA256})
    write(
        journal.root,
        "fixtures-manifest.json",
        {
            "schema": 1,
            "config_sha256": CONFIG_SHA256,
            "cases": [case],
            "source": str(source),
            "source_manifest_sha256": file_sha256(source_root / "fixtures-manifest.json"),
        },
    )
    return case, copied


def episode_succeeded(result):
    if result["failed"] or result.get("fatal") or not 1 <= len(result["actions"]) <= 3:
        return False
    last = result["actions"][-1]
    d = last["diagnostics"]
    return bool(
        last["execution_valid"]
        and d.get("settled") is True
        and last["coverage_after"] is not None
        and last["coverage_after"] >= 0.98
        and d.get("final_table_containment") == 1.0
        and d.get("final_rms_speed_m_s", float("inf")) <= 0.01
        and d.get("terminal_observation_valid") is not False
    )


def run_episode(env, journal, case, fixture, name, *, predict=None, epsilon=1.0):
    journal.check(reserve=journal.reserve + 3 * 180)
    result = pilot.episode(
        env,
        journal,
        case,
        fixture,
        name,
        predict=predict,
        epsilon=epsilon,
        seed=1001 + len(journal.document["episodes"]),
        coverage_stop=0.98,
        require_containment=True,
        reserve_seconds=journal.reserve,
    )
    result["success"] = episode_succeeded(result)
    result["kind"] = journal.kind
    write(journal.root / "episodes" / name, "result.json", result)
    journal.document["episodes"].append(name)
    journal.save()
    return result


def fit_checkpoint(journal, datasets, checkpoint, previous):
    from cloth_fling.learning import fit

    for folder in datasets:
        result = read(folder.parent, "result.json")
        if result["kind"] != "collection" or result["case"] != FIXTURE_ID:
            raise ValueError("check/confirmation or unrelated data cannot enter fitting")
        if file_sha256(folder / "manifest.json") != result["dataset_manifest_sha256"]:
            raise ValueError("training source manifest changed")
    return fit(datasets, checkpoint, previous=previous, steps=2000, check_deadline=journal.check)


def develop(
    env,
    journal,
    case,
    fixture,
    *,
    make_predictor=None,
    fit_fn=fit_checkpoint,
    episode_fn=run_episode,
):
    """Fixed schedule, first greedy success freezes weights; no adaptive extensions."""
    if make_predictor is None:
        from cloth_fling.learning import predictor

        make_predictor = predictor
    datasets, checkpoint, predict, candidate = [], None, None, None
    for stage, kind, count in schedule():
        with journal.operation(stage, kind) as outputs:
            if kind == "fit":
                previous = checkpoint
                checkpoint = journal.root / "checkpoints" / f"{stage}.pt"
                with _reservation_deadline(
                    time.monotonic() + journal.remaining() - journal.reserve
                ):
                    fit_fn(journal, datasets, checkpoint, previous)
                    predict = make_predictor(checkpoint)
                journal.seal(outputs, checkpoint)
                journal.seal(outputs, checkpoint.with_suffix(".json"))
                continue
            results = []
            for i in range(count):
                journal.check()
                name = f"{stage}-{i:02d}"
                result = episode_fn(
                    env,
                    journal,
                    case,
                    fixture,
                    name,
                    predict=predict,
                    epsilon=0.0 if kind == "check" else (1.0 if checkpoint is None else 0.5),
                )
                results.append(result)
                journal.seal(outputs, journal.root / "episodes" / name / "result.json")
                journal.seal(outputs, journal.root / "episodes" / name / "cleanup.json")
                journal.seal(outputs, journal.root / "episodes" / name / "restoration.json")
                if kind == "collection":
                    datasets.append(journal.root / "episodes" / name / "dataset")
            write(journal.root, f"{stage}-results.json", {"episodes": results})
            journal.seal(outputs, journal.root / f"{stage}-results.json")
            if kind == "check" and results[0]["success"]:
                candidate = results[0]
        if candidate:
            break
    if candidate is None:
        return "not_qualified"
    selection = {
        "checkpoint": str(checkpoint.relative_to(journal.root)),
        "weights_sha256": file_sha256(checkpoint),
        "greedy_check": candidate["episode"],
    }
    claim(journal.root / "selection.json", selection)
    for index in range(2):
        with journal.operation(f"confirmation-{index:02d}", "confirmation") as outputs:
            if file_sha256(checkpoint) != selection["weights_sha256"]:
                raise ValueError("frozen confirmation checkpoint changed")
            result = episode_fn(
                env,
                journal,
                case,
                fixture,
                f"confirmation-{index:02d}",
                predict=predict,
                epsilon=0.0,
            )
            folder = journal.root / "episodes" / result["episode"]
            for name in ("result.json", "cleanup.json", "restoration.json"):
                journal.seal(outputs, folder / name)
        if not result["success"]:
            return "confirmation_failed"
    return "qualified"


def episode_results(root):
    results = []
    for path in sorted((Path(root) / "episodes").glob("*/result.json")):
        result = json.loads(path.read_text())
        # The shared collector writes its receipt before the campaign annotates it.
        # A crash in that interval is preserved as an uncertain, non-qualifying episode.
        result.setdefault("kind", "uncertain")
        result.setdefault("success", False)
        results.append(result)
    return results


def render_evidence(journal):
    from cloth_fling.replay import render_replay

    results = episode_results(journal.root)
    checks = [r for r in results if r["kind"] == "check"]
    if (journal.root / "selection.json").exists():
        selected = read(journal.root, "selection.json")["greedy_check"]
        chosen = [r for r in results if r["episode"] == selected or r["kind"] == "confirmation"]
    elif checks:
        chosen = [max(checks, key=lambda r: r["score"])]
    else:
        chosen = [max(results, key=lambda r: r["score"])] if results else []
    videos = []
    for result in chosen:
        if not result["actions"]:
            continue
        journal.check()
        source = journal.root / "episodes" / result["episode"] / "dataset"
        target = journal.root / "replays" / result["episode"]
        with _reservation_deadline(time.monotonic() + min(300, journal.remaining())):
            render_replay(source, target, fps=10, max_frames=2000)
        for name in ("combined.mp4", "overhead.mp4", "oblique.mp4"):
            path = target / name
            videos.append(
                {"path": str(path.relative_to(journal.root)), "sha256": file_sha256(path)}
            )
    write(journal.root, "replays.json", videos)


def report(root, *, verify=True):
    """Read-only verification; callers may persist the returned report separately."""
    root = Path(root)
    ledger = read(root, "ledger.json")
    errors = []
    intents = [json.loads(p.read_text()) for p in sorted((root / "action-intents").glob("*.json"))]
    if intents != ledger["actions"]:
        errors.append("action intent/ledger mismatch")
    stage_intents = {p.stem for p in (root / "stage-intents").glob("*.json")}
    if stage_intents != {s["stage"] for s in ledger["stages"]}:
        errors.append("stage intent/ledger mismatch")
    results = episode_results(root)
    expected_episodes = set(ledger["episodes"])
    if {r["episode"] for r in results} != expected_episodes:
        errors.append("episode result/ledger mismatch")
    if len(list((root / "episodes").glob("*/intent.json"))) != len(results):
        errors.append("uncertain episode intent")
    if sum(len(r["actions"]) for r in results) != len(intents):
        errors.append("action receipt mismatch")
    if max(len(intents), len(ledger["actions"])) > ACTION_LIMIT:
        errors.append("action limit exceeded")
    if ledger["charged_seconds"] > WALL_SECONDS:
        errors.append("wall limit exceeded")
    if verify:
        if read(root, "protocol.json") != PROTOCOL:
            errors.append("protocol changed")
        for stage in ledger["stages"]:
            if stage["status"] != "complete":
                errors.append(f"uncertain stage: {stage['stage']}")
            for name, expected in stage["outputs"].items():
                path = relative_path(root, name)
                if not path.exists() or file_sha256(path) != expected:
                    errors.append(f"sealed output changed: {name}")
        for result in results:
            folder = root / "episodes" / result["episode"]
            dataset = folder / "dataset"
            if file_sha256(dataset / "manifest.json") != result["dataset_manifest_sha256"]:
                errors.append(f"dataset manifest changed: {result['episode']}")
            if not verify_dataset(dataset)["valid"]:
                errors.append(f"dataset verification: {result['episode']}")
            if not read(folder, "cleanup.json")["passed"]:
                errors.append(f"cleanup: {result['episode']}")
            if read(folder, "restoration.json") != {"passed": True, "state_hash": STATE_HASH}:
                errors.append(f"restoration: {result['episode']}")
            if result["success"] != episode_succeeded(result):
                errors.append(f"success inconsistent: {result['episode']}")
            if result["success"]:
                rows = read_transitions(dataset).to_pylist()
                native = read_support(dataset, rows[-1]["final_state_path"])
                config = scene_config_from_dict(json.loads(native["config_json"]))
                import numpy as np

                p, v = native["positions"], native["velocities"]
                xmin, xmax, ymin, ymax = config.table_bounds_m
                contained = (
                    (p[:, 0] >= xmin) & (p[:, 0] <= xmax) & (p[:, 1] >= ymin) & (p[:, 1] <= ymax)
                ).all()
                if (
                    not contained
                    or not native["free"].all()
                    or np.sqrt(np.mean(np.sum(v * v, axis=1))) > 0.01
                ):
                    errors.append(f"native success state: {result['episode']}")
        if (root / "scene-cleanup.json").exists() and not read(root, "scene-cleanup.json")[
            "passed"
        ]:
            errors.append("scene cleanup failed")
        if (root / "selection.json").exists():
            selected = read(root, "selection.json")
            if (
                file_sha256(relative_path(root, selected["checkpoint"]))
                != selected["weights_sha256"]
            ):
                errors.append("selected weights changed")
        for video in read(root, "replays.json") if (root / "replays.json").exists() else []:
            if file_sha256(relative_path(root, video["path"])) != video["sha256"]:
                errors.append("replay changed")
    checks = [r for r in results if r["kind"] == "check"]
    confirmations = [r for r in results if r["kind"] == "confirmation"]
    qualified = bool(
        verify
        and not errors
        and ledger["status"] == "complete"
        and ledger["outcome"] == "qualified"
        and any(r["success"] for r in checks)
        and len(confirmations) == 2
        and all(r["success"] for r in confirmations)
        and (root / "scene-cleanup.json").exists()
        and (root / "replays.json").exists()
    )
    return {
        "status": ledger["status"],
        "outcome": ledger["outcome"],
        "qualified": qualified,
        "verification_completed": verify,
        "integrity_errors": errors,
        "claimed_actions": max(len(intents), len(ledger["actions"])),
        "completed_episodes": len(results),
        "charged_seconds": ledger["charged_seconds"],
        "coverage_target": 0.98,
        "maximum_actions_per_episode": 3,
        "best_greedy_coverage": max((r["score"] for r in checks), default=None),
        "maximum_material_edge_ratio": max(
            (a["diagnostics"].get("max_edge_ratio", 0) for r in results for a in r["actions"]),
            default=None,
        ),
        "generalization_evaluated": False,
        "material_qualification": False,
        "episodes": [
            {
                "episode": r["episode"],
                "kind": r["kind"],
                "success": r["success"],
                "failed": r["failed"],
                "coverage": [a["coverage_after"] for a in r["actions"]],
                "predicted_gain": [a.get("predicted_gain") for a in r["actions"]],
                "measured_gain": [a["coverage_gain"] for a in r["actions"]],
            }
            for r in results
        ],
    }


def run(root, *, backend_factory=None):
    journal = Journal(root)  # Existing roots fail before any lease, scene or reporting mutation.
    env = None
    error = None
    try:
        with reference_lease(journal.root):
            try:
                with journal.operation("prepare", "prepare") as outputs:
                    with _reservation_deadline(time.monotonic() + min(180, journal.remaining())):
                        for name, expected in frozen_laydown_spec()[
                            "implementation_sha256"
                        ].items():
                            if file_sha256(Path(__file__).parent / name) != expected:
                                raise ValueError(f"frozen implementation changed: {name}")
                        case, fixture = prepare_fixture(journal)
                        if backend_factory is None:
                            from cloth_fling.sim.reference import ReferenceBackend

                            backend_factory = ReferenceBackend
                        env = backend_factory(
                            scene_config_from_dict(frozen_laydown_spec()["scene"])
                        )
                        observation = pilot.restore_exact(env, fixture)
                        metrics = pilot.state_metrics(env)
                        if (
                            not metrics["contained"]
                            or not metrics["native_free"]
                            or metrics["rms_speed_m_s"] > 0.01
                        ):
                            raise ValueError("frozen fixture acceptance changed")
                        if not pilot.build_action_batch(observation, env.config).feasible.any():
                            raise ValueError("frozen fixture has no feasible actions")
                        write(
                            journal.root,
                            "restoration.json",
                            {"passed": True, "state_hash": STATE_HASH},
                        )
                        for path in (journal.root / "fixture").iterdir():
                            journal.seal(outputs, path)
                        for name in (
                            "protocol.json",
                            "implementation.json",
                            "frozen-laydown.json",
                            "fixtures-manifest.json",
                            "restoration.json",
                        ):
                            journal.seal(outputs, journal.root / name)
                journal.document["outcome"] = develop(env, journal, case, fixture)
            finally:
                if env is not None:
                    certificate = certify_release(env)
                    write(journal.root, "scene-cleanup.json", certificate)
                    env.close()
                    if not certificate["passed"]:
                        raise RuntimeError("native scene cleanup failed")
            with journal.operation("replays", "replays") as outputs:
                render_evidence(journal)
                journal.seal(outputs, journal.root / "replays.json")
            journal.document["status"] = "complete"
    except BaseException as exc:
        error = exc
        journal.document.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, TimeoutError):
            journal.document["outcome"] = "budget_exhausted"
    finally:
        journal.save()
    if isinstance(error, TimeoutError) and journal.remaining() > 300:
        # Reconstruct existing evidence within the reserved hour; never resume physics.
        try:
            with reference_lease(journal.root), journal.operation("replays", "replays") as outputs:
                render_evidence(journal)
                journal.seal(outputs, journal.root / "replays.json")
        except Exception as exc:
            journal.document["replay_error"] = f"{type(exc).__name__}: {exc}"
            journal.save()
    # Cleanup is already durable, even if reporting itself fails or times out.
    try:
        journal.kind = "report"
        with _reservation_deadline(time.monotonic() + max(0.01, journal.remaining())):
            result = report(journal.root)
    except Exception as exc:
        journal.document.update(status="incomplete", reporting_error=f"{type(exc).__name__}: {exc}")
        journal.save()
        result = report(journal.root, verify=False)
    journal.save()
    result["charged_seconds"] = journal.document["charged_seconds"]
    if result["charged_seconds"] > WALL_SECONDS:
        result["qualified"] = False
        result["integrity_errors"].append("wall limit exceeded during reporting")
    write(journal.root, "overfit-report.json", result)
    if error:
        print(f"single-reset experiment stopped: {error}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = run(args.output) if args.command == "run" else report(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" and not result["integrity_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
