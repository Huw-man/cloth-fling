"""Bounded near-half-fold RGB pilot. Fresh roots only; no physical resume."""

import argparse
import json
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

from cloth_fling import learning_pilot as pilot
from cloth_fling.benchmark_run import _reservation_deadline
from cloth_fling.contracts import scene_config_from_dict
from cloth_fling.data import read_support, read_transitions, verify_dataset
from cloth_fling.data.storage import file_sha256, relative_path
from cloth_fling.frozen_laydown import CONFIG_SHA256, frozen_laydown_spec
from cloth_fling.learning_overfit import action_reaches_target
from cloth_fling.provenance import runtime_provenance
from cloth_fling.reference_pilot import certify_release
from cloth_fling.reference_run import reference_lease

DEFAULT_OUTPUT = "runs/near-half-fold-pilot-v1"
STAGES = ("prepare", "exploration", "initial", "refresh", "refreshed", "validation", "test")
PROTOCOL = {
    "version": "near_half_fold_pilot_v1",
    "wall_seconds": 21600,
    "collection_reserve_seconds": 7200,
    "action_limit": 216,
    "fixture_limit": 12,
    "frozen_config_sha256": CONFIG_SHA256,
    "splits": {
        "train": {"fractions": [0.46, 0.50, 0.54], "angles_deg": [-10, 10]},
        "validation": {"fractions": [0.48], "angles_deg": [-5, 5]},
        "test": {"fractions": [0.47, 0.53], "angles_deg": [-7, 7]},
    },
    "crease_radius_m": 0.02,
    "episodes_per_training_fixture_per_round": 4,
    "evaluation_repeats": 2,
    "actions_per_episode": 3,
    "coverage_stop": 0.8,
    "require_containment": True,
    "refresh_epsilon": 0.5,
    "training_steps": 2000,
    "training_seed": 1001,
    "minimum_training_samples": 32,
    "old_labels_or_weights": False,
    "provisional_success_count": 6,
}
write, read, claim = pilot.write, pilot.read, pilot.claim


def fixture_cases():
    return [
        {
            "id": f"fold-{round(fraction * 100):02d}-{angle:+03d}",
            "family": f"fold-{round(fraction * 100):02d}",
            "fraction": fraction,
            "angle_deg": angle,
            "split": split,
        }
        for split, spec in PROTOCOL["splits"].items()
        for fraction in spec["fractions"]
        for angle in spec["angles_deg"]
    ]


def episode_schedule(stage, selected="initial"):
    """Balanced evaluation order, fixed seeds, no adaptive replacements."""
    split = "train" if stage in ("exploration", "refresh") else stage
    records = [c for c in fixture_cases() if c["split"] == split]
    schedule = []
    for index, case in enumerate(records):
        repeats = 4 if split == "train" else 2
        for repeat in range(repeats):
            policies = (
                ["uniform" if stage == "exploration" else "mixed"]
                if split == "train"
                else ["initial", "refreshed"]
                if split == "validation"
                else ["uniform", selected]
            )
            if split != "train" and (index + repeat) % 2:
                policies = list(reversed(policies))
            for policy in policies:
                schedule.append(
                    {
                        "case": case["id"],
                        "policy": policy,
                        "repeat": repeat,
                        "episode": f"{stage}-{case['id']}-{policy}-{repeat:02d}",
                        "seed": (
                            10000
                            if stage == "exploration"
                            else 20000
                            if stage == "refresh"
                            else 30000
                            if stage == "validation"
                            else 40000
                        )
                        + index * 10
                        + repeat,
                    }
                )
    return schedule


class Journal:
    def __init__(self, root, *, prior_seconds=0.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic() - prior_seconds
        self.stage = "prepare"
        self.document = {
            "status": "running_never_resume",
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

    @property
    def reserve(self):
        return (
            PROTOCOL["collection_reserve_seconds"]
            if self.stage in ("prepare", "exploration", "refresh")
            else 0
        )

    def remaining(self):
        return PROTOCOL["wall_seconds"] - (time.monotonic() - self.started)

    def save(self):
        self.document["charged_seconds"] = time.monotonic() - self.started
        write(self.root, "ledger.json", self.document)

    def check(self, *, reserve=None):
        if self.remaining() <= (self.reserve if reserve is None else reserve):
            raise TimeoutError("family pilot deadline or reserved final budget reached")

    def claim_action(self, episode, step):
        self.check(reserve=self.reserve + 180)
        if episode == getattr(self, "carried_initialization", {}).get("episode"):
            raise RuntimeError("failed initialization slot already consumed; retry forbidden")
        scheduled = {x["episode"] for x in episode_schedule(self.stage)}
        # The selected checkpoint may be refreshed during test.
        if self.stage == "test":
            scheduled |= {x["episode"] for x in episode_schedule("test", "refreshed")}
        if episode not in scheduled or step not in range(3):
            raise RuntimeError("unscheduled action")
        if len(self.document["actions"]) >= PROTOCOL["action_limit"]:
            raise RuntimeError("family pilot action limit reached")
        if any(a["episode"] == episode and a["step"] == step for a in self.document["actions"]):
            raise RuntimeError("action already claimed; retry forbidden")
        intent = {
            "index": len(self.document["actions"]),
            "episode": episode,
            "step": step,
            "stage": self.stage,
            "status": "claimed_never_retry",
        }
        claim(self.root / "action-intents" / f"{intent['index']:04d}.json", intent)
        self.document["actions"].append(intent)
        self.save()

    def seal(self, outputs, path):
        path = Path(path)
        outputs[str(path.relative_to(self.root))] = file_sha256(path)

    @contextmanager
    def operation(self, stage):
        completed = [s["stage"] for s in self.document["stages"]]
        if completed != list(STAGES[: len(completed)]) or len(completed) >= len(STAGES):
            raise RuntimeError("invalid or terminal stage history")
        if stage != STAGES[len(completed)] or any(
            s["status"] != "complete" for s in self.document["stages"]
        ):
            raise RuntimeError("uncertain or out-of-order stage; resume forbidden")
        self.stage = stage
        self.check()
        if read(self.root, "protocol.json") != PROTOCOL:
            raise ValueError("protocol changed")
        if read(self.root, "implementation.json") != pilot.implementation_identity():
            raise ValueError("implementation changed")
        for entry in self.document["stages"]:
            for name, expected in entry["outputs"].items():
                if file_sha256(relative_path(self.root, name)) != expected:
                    raise ValueError(f"sealed artifact changed: {name}")
        intents = [
            json.loads(p.read_text()) for p in sorted((self.root / "action-intents").glob("*.json"))
        ]
        if intents != self.document["actions"]:
            raise RuntimeError("uncertain action intent/ledger gap")
        claim(self.root / "stage-intents" / f"{stage}.json", {"stage": stage})
        entry = {"stage": stage, "status": "claimed_never_retry", "outputs": {}}
        self.document["stages"].append(entry)
        self.save()
        print(f"stage {stage}: {len(intents)}/216 actions claimed", flush=True)
        try:
            # collect_episodes owns ITIMER_REAL during reset, selection and execution.
            # Only non-collection stages may hold an enclosing watchdog.
            watchdog = (
                _reservation_deadline(time.monotonic() + self.remaining() - self.reserve)
                if stage in ("prepare", "initial", "refreshed")
                else nullcontext()
            )
            with watchdog:
                yield entry["outputs"]
                self.check()
            entry["status"] = "complete"
        except BaseException as exc:
            entry.update(status="failed_or_uncertain", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.save()


def fixtures(root):
    return pilot.fixtures(root, expected_cases=fixture_cases())


def success(result):
    if result.get("failed") or result.get("fatal"):
        return False
    actions = result["actions"]
    if actions:
        return action_reaches_target(actions[-1], PROTOCOL["coverage_stop"])
    return result["collection"]["episode_reports"][0].get("lifecycle") == "initially_unfolded"


def policy_metrics(results):
    successes = [r for r in results if success(r)]
    return {
        "episodes": len(results),
        "successes": len(successes),
        "success_rate": len(successes) / len(results) if results else None,
        "success_within_actions": {
            str(n): sum(len(r["actions"]) <= n for r in successes) for n in (1, 2, 3)
        },
        "mean_failure_aware_terminal_coverage": float(np.mean([r["score"] for r in results]))
        if results
        else None,
        "failed_episodes": sum(r["failed"] for r in results),
        "mean_actions_on_success": float(np.mean([len(r["actions"]) for r in successes]))
        if successes
        else None,
    }


def choose_checkpoint(results):
    metrics = {
        name: policy_metrics([r for r in results if r["policy"] == name])
        for name in ("initial", "refreshed")
    }
    if any(m["episodes"] != 4 for m in metrics.values()):
        raise ValueError("validation schedule incomplete")
    selected = max(
        metrics,
        key=lambda name: (
            metrics[name]["successes"],
            metrics[name]["mean_failure_aware_terminal_coverage"],
            name == "initial",
        ),
    )
    return selected, metrics


def run_episodes(env, journal, stage, outputs):
    from cloth_fling.learning import predictor

    with _reservation_deadline(time.monotonic() + journal.remaining() - journal.reserve):
        records, states = fixtures(journal.root)
        selected = (
            read(journal.root, "selection.json")["checkpoint"] if stage == "test" else "initial"
        )
        if (
            stage == "test"
            and file_sha256(journal.root / "checkpoints" / f"{selected}.pt")
            != read(journal.root, "selection.json")["weights_sha256"]
        ):
            raise ValueError("selected checkpoint changed")
        jobs = episode_schedule(stage, selected)
        names = {"initial" if j["policy"] == "mixed" else j["policy"] for j in jobs} - {"uniform"}
        predictions = {n: predictor(journal.root / "checkpoints" / f"{n}.pt") for n in names}
        cases = {c["id"]: c for c in records}
    results = []
    for job in jobs:
        journal.check()
        carried = getattr(journal, "carried_initialization", None)
        if carried and job["episode"] == carried["episode"]:
            result = dict(
                carried,
                policy=job["policy"],
                repeat=job["repeat"],
                success=False,
                carried_failed_initialization=True,
            )
            folder = journal.root / "episodes" / job["episode"]
            write(folder, "result.json", result)
            journal.document["episodes"].append(
                {
                    **job,
                    "result_sha256": file_sha256(folder / "result.json"),
                    "carried_failed_initialization": True,
                }
            )
            journal.save()
            journal.seal(outputs, folder / "result.json")
            journal.seal(outputs, folder / "dataset" / "manifest.json")
            results.append(result)
            continue
        name = job["policy"]
        result = pilot.episode(
            env,
            journal,
            cases[job["case"]],
            states[job["case"]],
            job["episode"],
            predict=predictions.get("initial" if name == "mixed" else name),
            epsilon=1.0 if name == "uniform" else 0.5 if name == "mixed" else 0.0,
            seed=job["seed"],
            coverage_stop=0.8,
            require_containment=True,
            reserve_seconds=journal.reserve,
        )
        result.update(policy=name, repeat=job["repeat"], success=success(result))
        folder = journal.root / "episodes" / job["episode"]
        write(folder, "result.json", result)
        journal.document["episodes"].append(
            {**job, "result_sha256": file_sha256(folder / "result.json")}
        )
        journal.save()
        journal.seal(outputs, folder / "result.json")
        journal.seal(outputs, folder / "dataset" / "manifest.json")
        results.append(result)
    write(journal.root, f"{stage}-results.json", {"episodes": results})
    journal.seal(outputs, journal.root / f"{stage}-results.json")
    if stage == "validation":
        selected, metrics = choose_checkpoint(results)
        claim(
            journal.root / "selection.json",
            {
                "checkpoint": selected,
                "validation_metrics": metrics,
                "weights_sha256": file_sha256(journal.root / "checkpoints" / f"{selected}.pt"),
            },
        )
        journal.seal(outputs, journal.root / "selection.json")


def training_datasets(journal, stage):
    fixtures(journal.root)
    datasets = []
    expected = {c["id"] for c in fixture_cases() if c["split"] == "train"}
    for source in ["exploration"] + (["refresh"] if stage == "refreshed" else []):
        results = read(journal.root, f"{source}-results.json")["episodes"]
        if [r["episode"] for r in results] != [j["episode"] for j in episode_schedule(source)]:
            raise ValueError("training episode schedule changed")
        for result in results:
            path = journal.root / "episodes" / result["episode"] / "dataset"
            if result["case"] not in expected or result["split"] != "train":
                raise ValueError("non-training fixture in training sources")
            if file_sha256(path / "manifest.json") != result["dataset_manifest_sha256"]:
                raise ValueError("training dataset changed")
            if not verify_dataset(path)["valid"]:
                raise ValueError("training artifact verification failed")
            datasets.append(path)
    return datasets


def fit_stage(journal, stage, outputs):
    from cloth_fling.learning import fit

    result = fit(
        training_datasets(journal, stage),
        journal.root / "checkpoints" / f"{stage}.pt",
        previous=journal.root / "checkpoints" / "initial.pt" if stage == "refreshed" else None,
        steps=PROTOCOL["training_steps"],
        check_deadline=journal.check,
    )
    write(journal.root, f"{stage}-training.json", result)
    for name in (f"{stage}-training.json", f"checkpoints/{stage}.pt", f"checkpoints/{stage}.json"):
        journal.seal(outputs, journal.root / name)


def render_evidence(journal):
    """Saved-geometry replay only, including a terminal campaign's available failure."""
    from cloth_fling.replay import render_replay

    results = [
        json.loads(p.read_text()) for p in sorted((journal.root / "episodes").glob("*/result.json"))
    ]
    selected = []
    tests = [r for r in results if r.get("split") == "test" and r.get("policy") != "uniform"]
    for pool, key in (
        (tests or results, lambda r: r["score"]),
        ([r for r in results if r["failed"]], lambda r: len(r["actions"])),
    ):
        if pool:
            selected.append(max(pool, key=key)["episode"])
    # Preserve the most recent attempted episode even if post-action auditing failed.
    if journal.document["actions"]:
        selected.append(journal.document["actions"][-1]["episode"])
    videos = []
    for name in dict.fromkeys(selected):
        source = journal.root / "episodes" / name / "dataset"
        if not (source / "manifest.json").exists() or not read_transitions(source).num_rows:
            continue
        journal.check(reserve=180)
        target = journal.root / "replays" / name
        with _reservation_deadline(time.monotonic() + min(180, journal.remaining())):
            render_replay(source, target, fps=10, max_frames=2000)
        videos.append(
            {
                "episode": name,
                "relative_file": f"replays/{name}/combined.mp4",
                "sha256": file_sha256(target / "combined.mp4"),
            }
        )
    write(journal.root, "replays.json", videos)


def report(root, *, verify=True):
    """Offline audit includes unfinalized episodes; never resumes native actions."""
    root = Path(root)
    ledger = read(root, "ledger.json")
    errors = []
    if read(root, "protocol.json") != PROTOCOL:
        errors.append("protocol mismatch")
    intents = [json.loads(p.read_text()) for p in sorted((root / "action-intents").glob("*.json"))]
    if intents != ledger["actions"]:
        errors.append("action intent/ledger gap")
    for stage in ledger["stages"]:
        if stage["status"] != "complete":
            errors.append(f"uncertain stage: {stage['stage']}")
        for name, expected in stage["outputs"].items():
            if not (root / name).exists() or file_sha256(root / name) != expected:
                errors.append(f"sealed output changed: {name}")
    receipts = {e["episode"]: e for e in ledger["episodes"]}
    results, summaries, seen, strains = [], [], [], []
    rows_count = 0
    negative = 0
    for folder in sorted((root / "episodes").glob("*")):
        if not folder.is_dir():
            continue
        dataset = folder / "dataset"
        final = folder / "result.json"
        result = json.loads(final.read_text()) if final.exists() else None
        receipt = receipts.get(folder.name)
        if not receipt or not final.exists() or file_sha256(final) != receipt["result_sha256"]:
            errors.append(f"uncertain episode receipt: {folder.name}")
        rows = []
        reset_errors = []
        if (dataset / "manifest.json").exists():
            manifest = read(dataset, "manifest.json")
            for episode in manifest.get("episodes", []):
                if episode.get("reset_evidence"):
                    support = read_support(dataset, episode["reset_evidence"]["path"])
                    if support.get("reset_error"):
                        reset_errors.append(support["reset_error"])
            if verify and not verify_dataset(dataset)["valid"]:
                errors.append(f"dataset verification: {folder.name}")
            rows = read_transitions(dataset).to_pylist()
            if (
                result
                and file_sha256(dataset / "manifest.json") != result["dataset_manifest_sha256"]
            ):
                errors.append(f"dataset identity: {folder.name}")
        else:
            errors.append(f"missing dataset: {folder.name}")
        seen.extend((folder.name, i) for i in range(len(rows)))
        rows_count += len(rows)
        negative += sum(r.get("coverage_gain") is not None and r["coverage_gain"] < 0 for r in rows)
        for row in rows:
            d = read_support(dataset, row["evidence_path"]).get("diagnostics", {})
            if d.get("max_edge_ratio") is not None:
                strains.append(d["max_edge_ratio"])
        if result:
            results.append(result)
            if len(result["actions"]) != len(rows):
                errors.append(f"result/row mismatch: {folder.name}")
        summaries.append(
            {
                "episode": folder.name,
                "finalized": receipt is not None,
                "policy": result.get("policy") if result else None,
                "case": result.get("case") if result else None,
                "repeat": result.get("repeat") if result else None,
                "success": success(result) if result else False,
                "failed": result["failed"] if result else True,
                "failure_aware_terminal_coverage": result["score"] if result else 0,
                "coverage": [r.get("coverage_after") for r in rows],
                "reasons": [r.get("reason") for r in rows],
                "reset_errors": reset_errors,
                "actions": len(rows),
                "carried_failed_initialization": bool(
                    result and result.get("carried_failed_initialization")
                ),
            }
        )
    claimed = [(a["episode"], a["step"]) for a in intents]
    if sorted(claimed) != sorted(seen) or len(set(claimed)) != len(claimed):
        errors.append("physical action intent/row mismatch")
    if (
        len(intents) > PROTOCOL["action_limit"]
        or ledger["charged_seconds"] > PROTOCOL["wall_seconds"]
    ):
        errors.append("budget exceeded")
    for video in read(root, "replays.json") if (root / "replays.json").exists() else []:
        path = relative_path(root, video["relative_file"])
        if not path.exists() or file_sha256(path) != video["sha256"]:
            errors.append(f"replay identity: {video['relative_file']}")
    if verify and (root / "fixtures-manifest.json").exists():
        try:
            fixtures(root)
        except Exception as exc:
            errors.append(f"fixture verification: {exc}")
    cleanup = read(root, "scene-cleanup.json") if (root / "scene-cleanup.json").exists() else {}
    if not cleanup.get("passed"):
        errors.append("scene cleanup not certified")
    complete = bool(
        verify
        and ledger["status"] == "complete"
        and not errors
        and [s["stage"] for s in ledger["stages"]] == list(STAGES)
        and len(results) == 72
    )
    tests = [r for r in results if r["split"] == "test"]
    learned = policy_metrics([r for r in tests if r.get("policy") != "uniform"])
    uniform = policy_metrics([r for r in tests if r.get("policy") == "uniform"])
    result = {
        "status": ledger["status"],
        "pipeline_complete": complete,
        "verification_completed": verify,
        "error": ledger.get("error"),
        "charged_seconds": ledger["charged_seconds"],
        "claimed_actions": len(intents),
        "dataset_rows": rows_count,
        "negative_gains": negative,
        "integrity_errors": errors,
        "episodes": summaries,
        "learned_test": learned,
        "random_test": uniform,
        "provisional_scale_signal": bool(
            complete
            and learned["episodes"] == uniform["episodes"] == 8
            and learned["successes"] >= 6
            and learned["successes"] > uniform["successes"]
        ),
        "maximum_material_edge_ratio": max(strains, default=None),
        "material_qualification": False,
        "broad_generalization_evaluated": False,
        "replays": read(root, "replays.json") if (root / "replays.json").exists() else [],
    }
    return result


def validate_preaction_recovery(source, *, transfer_target=None):
    """Only a proved pre-restore timer collision can carry forward without physical retry."""
    source = Path(source)
    transfer = source / "preaction-transfer-intent.json"
    if transfer.exists() and (
        transfer_target is None
        or read(source, transfer.name)["target"] != str(Path(transfer_target).resolve())
    ):
        raise ValueError("prepared run already transferred; retry forbidden")
    ledger = read(source, "ledger.json")
    if (
        read(source, "protocol.json") != PROTOCOL
        or ledger["status"] != "incomplete"
        or ledger["actions"]
        or ledger["episodes"]
        or list((source / "action-intents").glob("*.json"))
        or [s["stage"] for s in ledger["stages"]] != ["prepare", "exploration"]
        or ledger["stages"][0]["status"] != "complete"
        or ledger["stages"][1]["status"] != "failed_or_uncertain"
    ):
        raise ValueError("recovery requires a terminal zero-action preparation-only run")
    expected = episode_schedule("exploration")[0]["episode"]
    directories = list((source / "episodes").iterdir())
    if len(directories) != 1 or directories[0].name != expected:
        raise ValueError("recovery may consume only the first failed initialization")
    folder = directories[0]
    result = read(folder, "result.json")
    if (
        result["actions"]
        or not result["failed"]
        or (folder / "restoration.json").exists()
        or list(folder.glob("action-*"))
        or not read(folder, "cleanup.json")["passed"]
        or not read(source, "scene-cleanup.json")["passed"]
        or read(source, "gpu-lease.json")["active"]
    ):
        raise ValueError("native action/restoration or uncertified cleanup forbids recovery")
    dataset = folder / "dataset"
    manifest = read(dataset, "manifest.json")
    if (
        not verify_dataset(dataset)["valid"]
        or read_transitions(dataset).num_rows
        or len(manifest["episodes"]) != 1
        or file_sha256(dataset / "manifest.json") != result["dataset_manifest_sha256"]
    ):
        raise ValueError("recovery dataset identity or empty-action proof failed")
    evidence = read_support(dataset, manifest["episodes"][0]["reset_evidence"]["path"])
    if (
        evidence.get("reset_error")
        != "RuntimeError: collection requires an unused process real-time timer"
    ):
        raise ValueError("only the identified pre-restoration timer conflict is recoverable")
    fixtures(source)
    for name, digest in ledger["stages"][0]["outputs"].items():
        if file_sha256(relative_path(source, name)) != digest:
            raise ValueError("prepared source identity changed")
    charged = ledger["charged_seconds"]
    if not 0 <= charged < PROTOCOL["wall_seconds"] - PROTOCOL["collection_reserve_seconds"]:
        raise ValueError("no authorized collection budget remains")
    return {
        "source": str(source.resolve()),
        "ledger_sha256": file_sha256(source / "ledger.json"),
        "manifest_sha256": file_sha256(source / "fixtures-manifest.json"),
        "prior_charged_seconds": charged,
        "consumed_episode": expected,
        "prior_action_count": 0,
        "fixture_preparations_carried": 12,
        "source_unchanged_and_terminal": True,
    }


def import_prepared(journal, source, recovery):
    # No preparation simulation, rerolls or repair of old receipts. Copy certified bytes.
    if validate_preaction_recovery(source, transfer_target=journal.root) != recovery:
        raise ValueError("pre-action recovery source changed")
    shutil.copytree(source / "fixtures", journal.root / "fixtures")
    shutil.copy2(source / "fixtures-manifest.json", journal.root / "fixtures-manifest.json")
    name = recovery["consumed_episode"]
    shutil.copytree(source / "episodes" / name, journal.root / "episodes" / name)
    journal.carried_initialization = read(source / "episodes" / name, "result.json")
    write(journal.root, "preaction-recovery.json", recovery)
    journal.document["preaction_recovery"] = recovery
    journal.save()


def run(root, *, backend_factory=None, prepared_from=None):
    recovery = validate_preaction_recovery(prepared_from) if prepared_from is not None else None
    journal = Journal(
        root, prior_seconds=recovery["prior_charged_seconds"] if recovery else 0.0
    )  # Existing/uncertain roots fail before lease/backend construction.
    env = None
    try:
        if recovery:
            claim(
                Path(prepared_from) / "preaction-transfer-intent.json",
                {
                    "target": str(journal.root.resolve()),
                    "never_retry": True,
                    "source_ledger_sha256": recovery["ledger_sha256"],
                },
            )
        with reference_lease(journal.root):
            try:
                with journal.operation("prepare") as outputs:
                    for name, expected in frozen_laydown_spec()["implementation_sha256"].items():
                        if file_sha256(Path(__file__).parent / name) != expected:
                            raise ValueError(f"frozen implementation changed: {name}")
                    if backend_factory is None:
                        from cloth_fling.sim.reference import ReferenceBackend

                        backend_factory = ReferenceBackend
                    with _reservation_deadline(
                        time.monotonic() + min(180, journal.remaining() - journal.reserve)
                    ):
                        env = backend_factory(
                            scene_config_from_dict(frozen_laydown_spec()["scene"])
                        )
                    if recovery:
                        import_prepared(journal, Path(prepared_from), recovery)
                    else:
                        pilot.prepare(env, journal, cases=fixture_cases())
                    for name in (
                        "protocol.json",
                        "implementation.json",
                        "frozen-laydown.json",
                        "fixtures-manifest.json",
                    ):
                        journal.seal(outputs, journal.root / name)
                    if recovery:
                        journal.seal(outputs, journal.root / "preaction-recovery.json")
                for stage in STAGES[1:]:
                    with journal.operation(stage) as outputs:
                        if stage in ("initial", "refreshed"):
                            fit_stage(journal, stage, outputs)
                        else:
                            run_episodes(env, journal, stage, outputs)
                journal.document["status"] = "complete"
            finally:
                if env is not None:
                    certificate = certify_release(env)
                    write(journal.root, "scene-cleanup.json", certificate)
                    env.close()
                    if not certificate["passed"]:
                        raise RuntimeError("scene cleanup failed")
    except BaseException as exc:
        journal.document.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        print(f"family pilot stopped: {type(exc).__name__}: {exc}", flush=True)
    finally:
        journal.save()
    # Reconstruction cannot continue physical actions, even after a fatal campaign stop.
    journal.stage = "replays"
    try:
        if journal.remaining() > 300:
            with reference_lease(journal.root):
                render_evidence(journal)
    except Exception as exc:
        journal.document.update(status="incomplete", replay_error=f"{type(exc).__name__}: {exc}")
    journal.save()
    try:
        with _reservation_deadline(time.monotonic() + max(0.01, journal.remaining())):
            result = report(journal.root)
    except Exception as exc:
        journal.document.update(status="incomplete", reporting_error=f"{type(exc).__name__}: {exc}")
        journal.save()
        result = report(journal.root, verify=False)
    journal.save()
    result["charged_seconds"] = journal.document["charged_seconds"]
    if journal.remaining() <= 0:
        result.update(pipeline_complete=False, provisional_scale_signal=False)
        result["integrity_errors"].append("deadline exceeded during closeout")
    write(journal.root, "family-report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--prepared-from",
        type=Path,
        help="Carry a verified zero-action timer failure into a fresh root; consume its episode slot",
    )
    args = parser.parse_args(argv)
    if args.command == "report" and args.prepared_from is not None:
        parser.error("--prepared-from applies only to run")
    result = (
        run(args.output, prepared_from=args.prepared_from)
        if args.command == "run"
        else report(args.output)
    )
    if args.command == "report":
        write(args.output, "family-report.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["pipeline_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
