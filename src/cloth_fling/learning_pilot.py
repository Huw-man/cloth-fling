"""One bounded, split-isolated RGB grasp-learning experiment; no physical resume."""

import argparse
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cloth_fling.benchmark_run import _reservation_deadline
from cloth_fling.collection import collect_episodes
from cloth_fling.contracts import scene_config_from_dict
from cloth_fling.data import read_support, read_transitions, verify_dataset
from cloth_fling.data.storage import atomic_write, file_sha256, json_bytes, relative_path
from cloth_fling.frozen_laydown import CONFIG_SHA256, frozen_laydown_spec
from cloth_fling.provenance import runtime_provenance
from cloth_fling.reference_diagonal_motion import campaign_stopping
from cloth_fling.reference_pilot import PilotExecutor, audit_tug_commands, certify_release
from cloth_fling.reference_run import reference_lease, save_bundle
from cloth_fling.reference_tune import read_hashed_bundle
from cloth_fling.sim.mesh import finite_radius_fold
from cloth_fling.spatial_actions import build_action_batch, choose_action

STAGES = (
    "prepare",
    "exploration",
    "initial",
    "refresh",
    "refreshed",
    "validation",
    "test",
    "replays",
)
LIMIT_SECONDS = 8 * 3600
COLLECTION_RESERVE_SECONDS = 2 * 3600
ACTION_LIMIT = 432
DEFAULT_OUTPUT = "runs/image-learning-pilot-v1"
PROTOCOL = {
    "version": "image_learning_pilot_v1",
    "wall_seconds": LIMIT_SECONDS,
    "action_limit": ACTION_LIMIT,
    "fixture_limit": 21,
    "frozen_config_sha256": CONFIG_SHA256,
    "fractions": {"train": [0.20, 0.35, 0.50], "validation": [0.30], "test": [0.25, 0.40, 0.45]},
    "angles_deg": [-15, 0, 15],
    "crease_radius_m": 0.02,
    "exploration_episodes_per_fixture": 8,
    "refresh_episodes_per_fixture": 5,
    "actions_per_episode": 3,
    "coverage_stop": 0.95,
    "refresh_epsilon": 0.5,
    "training_seed": 1001,
    "training_steps_per_round": 2000,
    "minimum_training_samples": 32,
    "behavior_gain": 0.05,
}
OUTPUTS = {
    "prepare": ["fixtures-manifest.json"],
    "exploration": ["exploration-results.json"],
    "initial": ["initial-training.json", "checkpoints/initial.pt", "checkpoints/initial.json"],
    "refresh": ["refresh-results.json"],
    "refreshed": [
        "refreshed-training.json",
        "checkpoints/refreshed.pt",
        "checkpoints/refreshed.json",
    ],
    "validation": ["validation-results.json", "selection.json"],
    "test": ["test-results.json"],
    "replays": ["replays.json"],
}


def implementation_identity():
    package = Path(__file__).parent
    return {str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))}


def write(root, name, value):
    atomic_write(Path(root), name, json_bytes(value))


def read(root, name):
    return json.loads(relative_path(Path(root), name).read_text())


def claim(path, value):
    """Exclusive, fsynced intent survives a crash before any native mutation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def initialize(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write(root, "protocol.json", PROTOCOL)
    write(root, "runtime.json", runtime_provenance())
    write(root, "implementation.json", implementation_identity())
    write(root, "frozen-laydown.json", frozen_laydown_spec())
    # The original spec is canonical JSON; also verify its implementation before GPU use.
    from cloth_fling import frozen_laydown

    for name, expected in frozen_laydown_spec()["implementation_sha256"].items():
        if file_sha256(Path(frozen_laydown.__file__).parent / name) != expected:
            raise ValueError(f"frozen implementation changed: {name}")
    write(
        root,
        "ledger.json",
        {"status": "ready", "charged_seconds": 0.0, "stages": [], "actions": []},
    )


class Journal:
    def __init__(self, root):
        self.root = Path(root)
        self.document = read(root, "ledger.json")
        if read(root, "protocol.json") != PROTOCOL:
            raise ValueError("pilot protocol changed")
        if read(root, "implementation.json") != implementation_identity():
            raise ValueError("pilot source changed after the experiment was claimed")
        intents = [
            read(root, str(p.relative_to(self.root)))
            for p in sorted((self.root / "action-intents").glob("*.json"))
        ]
        if intents != self.document["actions"]:
            raise RuntimeError("uncertain action intent/ledger gap; physical resume is forbidden")
        for stage in self.document["stages"]:
            if stage["status"] == "complete":
                for name, expected in stage["outputs"].items():
                    if file_sha256(self.root / name) != expected:
                        raise ValueError("completed stage artifact changed")
        self.started = None

    def save(self):
        write(self.root, "ledger.json", self.document)

    def remaining(self):
        active = time.monotonic() - self.started if self.started is not None else 0
        return LIMIT_SECONDS - self.document["charged_seconds"] - active

    def check(self, *, reserve=0):
        if self.remaining() <= reserve:
            raise TimeoutError("pilot wall budget exhausted or final-stage reserve reached")

    def claim_action(self, episode, step):
        self.check(
            reserve=COLLECTION_RESERVE_SECONDS if self.stage in ("exploration", "refresh") else 0
        )
        if len(self.document["actions"]) >= ACTION_LIMIT:
            raise RuntimeError("pilot action budget exhausted")
        index = len(self.document["actions"])
        intent = {
            "index": index,
            "episode": episode,
            "step": step,
            "stage": self.stage,
            "status": "claimed_never_retry",
        }
        claim(self.root / "action-intents" / f"{index:04d}.json", intent)
        self.document["actions"].append(intent)
        self.save()

    @contextmanager
    def operation(self, stage):
        if self.document["status"] not in ("ready", "between_stages"):
            raise RuntimeError("terminal or uncertain pilot cannot execute or resume")
        completed = [s["stage"] for s in self.document["stages"]]
        if completed != list(STAGES[: len(completed)]) or len(completed) >= len(STAGES):
            raise RuntimeError("invalid pilot stage history")
        if stage != STAGES[len(completed)]:
            raise RuntimeError("stage already claimed or prerequisite incomplete")
        self.check()
        claim(self.root / "stage-intents" / f"{stage}.json", {"stage": stage, "never_retry": True})
        self.stage = stage
        self.started = time.monotonic()
        entry = {"stage": stage, "status": "claimed_never_retry", "started_unix": time.time()}
        self.document["stages"].append(entry)
        self.document["status"] = "running_never_resume"
        self.save()
        try:
            yield
            self.check()
            entry["outputs"] = {p: file_sha256(self.root / p) for p in OUTPUTS[stage]}
        except BaseException as exc:
            entry.update(status="failed_or_uncertain", error=f"{type(exc).__name__}: {exc}")
            self.document["status"] = "incomplete"
            raise
        else:
            entry["status"] = "complete"
            self.document["status"] = "complete" if stage == STAGES[-1] else "between_stages"
        finally:
            entry["wall_seconds"] = time.monotonic() - self.started
            self.document["charged_seconds"] += entry["wall_seconds"]
            self.started = None
            self.save()


def folded_positions(rest, config, fraction, angle):
    """Reset-only finite-radius crease; never modifies material rest geometry."""
    result = np.asarray(rest).copy()
    lo, hi = result[:, 0].min(), result[:, 0].max()
    crease = hi - fraction * (hi - lo)
    result[:, 0] -= crease
    result = finite_radius_fold(result, 0.02, config.table_surface_z_m + 0.011)
    result[:, 0] += crease
    result[:, :2] -= (result[:, :2].min(axis=0) + result[:, :2].max(axis=0)) / 2
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    result[:, :2] = result[:, :2] @ rotation.T + np.asarray(config.table_center_xy_m)
    return result


def state_metrics(env):
    p, v = env.get_positions(), env.get_velocities()
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise RuntimeError("nonfinite native cloth state")
    xmin, xmax, ymin, ymax = env.config.table_bounds_m
    lengths = np.linalg.norm(p[env.edges[:, 1]] - p[env.edges[:, 0]], axis=1)
    ratio = float(np.max(lengths / env.rest_edge_lengths))
    if (
        ratio >= 3
        or np.any(np.abs(p[:, :2]) >= env.config.table_size_m)
        or np.any(p[:, 2] < -0.1)
        or np.any(p[:, 2] > 2)
    ):
        raise RuntimeError("native deformation/domain safeguard")
    return {
        "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(v**2, axis=1)))),
        "contained": bool(
            ((p[:, 0] >= xmin) & (p[:, 0] <= xmax) & (p[:, 1] >= ymin) & (p[:, 1] <= ymax)).all()
        ),
        "native_free": bool(env._free().all()),
        "max_edge_ratio": ratio,
    }


def restore_exact(env, state):
    observation = env.restore(state)
    actual = env.snapshot()
    if actual["state_hash"] != state["state_hash"] or not all(
        np.array_equal(actual[k], state[k])
        for k in ("positions", "velocities", "free", "faces", "rest_positions")
    ):
        raise RuntimeError("exact fixture restoration regression")
    return observation


def fixture_cases():
    return [
        {
            "id": f"fold-{round(fraction * 100):02d}-{angle:+03d}",
            "family": f"fold-{round(fraction * 100):02d}",
            "fraction": fraction,
            "angle_deg": angle,
            "split": split,
        }
        for split, fractions in PROTOCOL["fractions"].items()
        for fraction in fractions
        for angle in PROTOCOL["angles_deg"]
    ]


def prepare(env, journal, *, cases=None):
    root = journal.root
    records = []
    for case in fixture_cases() if cases is None else cases:
        journal.check(reserve=COLLECTION_RESERVE_SECONDS)
        folder = root / "fixtures" / case["id"]
        claim(folder / "intent.json", case)
        record = {**case, "accepted": False}
        try:
            with _reservation_deadline(
                time.monotonic() + min(180, journal.remaining() - COLLECTION_RESERVE_SECONDS)
            ):
                env.reset()
                initial = env.snapshot()
                save_bundle(folder, "native-time-zero", initial)
                positions = folded_positions(
                    env.rest_positions, env.config, case["fraction"], case["angle_deg"]
                )
                env.cloth.set_particles_pos(positions)
                env.cloth.set_particles_vel(np.zeros_like(positions))
                env.reset_metadata = {"kind": "learning_pilot_fold", **case}
                env._dirty()
                save_bundle(folder, "initialized", env.snapshot())
                stable = 0.0
                for _ in range(round(10 / env.config.controller_dt)):
                    journal.check(reserve=COLLECTION_RESERVE_SECONDS)
                    env.step(round(env.config.controller_dt / env.config.sim_dt))
                    metrics = state_metrics(env)
                    stable = (
                        stable + env.config.controller_dt
                        if (
                            metrics["native_free"]
                            and metrics["contained"]
                            and metrics["rms_speed_m_s"] <= 0.01
                        )
                        else 0.0
                    )
                    if stable >= 0.5 - 1e-9:
                        break
                save_bundle(folder, "termination", env.snapshot())
                if stable < 0.5 - 1e-9:
                    raise RuntimeError(
                        "fixture did not settle contained within ten simulated seconds"
                    )
                batch = build_action_batch(env.observe(), env.config)
                if not batch.feasible.any():
                    raise RuntimeError("fixture has no feasible image action")
                state = env.snapshot()
                ref = save_bundle(folder, "fixture", state)
                restore_exact(env, state)
                record.update(
                    accepted=True,
                    metrics=metrics,
                    state_hash=state["state_hash"],
                    fixture={
                        "path": f"fixtures/{case['id']}/{ref['path']}",
                        "sha256": ref["sha256"],
                    },
                    restoration_passed=True,
                )
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            try:
                save_bundle(folder, "failure", env.snapshot())
            except Exception as capture:
                record["capture_error"] = str(capture)
            raise
        finally:
            record["cleanup"] = certify_release(env)
            write(folder, "cleanup.json", record["cleanup"])
            write(folder, "preparation.json", record)
            records.append(record)
            write(root, "preparation-progress.json", {"cases": records})
        if not record["cleanup"]["passed"]:
            raise RuntimeError("fixture cleanup failure")
        print(f"prepared {case['id']}", flush=True)
    claim(
        root / "fixtures-manifest.json",
        {"schema": 1, "config_sha256": CONFIG_SHA256, "cases": records},
    )


def fixtures(root, *, expected_cases=None):
    prepare_entry = read(root, "ledger.json")["stages"][0]
    if prepare_entry["status"] != "complete" or prepare_entry["outputs"][
        "fixtures-manifest.json"
    ] != file_sha256(Path(root) / "fixtures-manifest.json"):
        raise ValueError("frozen fixture manifest changed or preparation is incomplete")
    manifest = read(root, "fixtures-manifest.json")
    expected = fixture_cases() if expected_cases is None else expected_cases
    records = manifest["cases"]
    if manifest["config_sha256"] != CONFIG_SHA256 or len(records) != len(expected):
        raise ValueError("incomplete or incompatible frozen fixture set")
    states, split_hashes = {}, {}
    for case, identity in zip(records, expected, strict=True):
        if (
            any(case[k] != identity[k] for k in identity)
            or not case["accepted"]
            or not case["cleanup"]["passed"]
        ):
            raise ValueError("fixture split or acceptance changed")
        ref = case["fixture"]
        path = relative_path(Path(root), ref["path"])
        if file_sha256(path) != ref["sha256"]:
            raise ValueError("fixture artifact corruption")
        state = read_hashed_bundle(path.parent, path.name)
        if state["state_hash"] != case["state_hash"] or not np.asarray(state["free"]).all():
            raise ValueError("fixture state identity changed")
        old = split_hashes.setdefault(state["state_hash"], case["split"])
        if old != case["split"]:
            raise ValueError("fixture state leaked across splits")
        states[case["id"]] = state
    return records, states


class EpisodePolicy:
    def __init__(self, predict=None, epsilon=1.0):
        self.predict, self.epsilon = predict, epsilon
        self.selection, self.fatal = None, None

    def __call__(self, observation, config, *, rng, **kwargs):
        try:
            batch = build_action_batch(observation, config)
        except ValueError as exc:
            if str(exc) != "no RGB cloth foreground":
                self.fatal = str(exc)
                raise
            return None
        except Exception as exc:
            self.fatal = f"rendering/action-batch regression: {exc}"
            raise
        try:
            values = None

            def measured_prediction(rgb):
                nonlocal values
                values = self.predict(rgb)
                return values

            prediction = (
                measured_prediction
                if self.predict is not None and rng.random() >= self.epsilon
                else None
            )
            action = choose_action(batch, rng=rng, predictor=prediction)
            if action is not None:
                self.selection = {
                    "encoding": action.encoding,
                    "valid_action_mask": batch.feasible,
                    "observation_id": observation.observation_id,
                    "predicted_gain": (
                        float(np.asarray(values)[tuple(action.encoding["map_index_tvu"])])
                        if values is not None
                        else None
                    ),
                }
            return action
        except Exception as exc:
            self.fatal = f"prediction/codec failure: {exc}"
            raise


class EpisodeExecutor:
    def __init__(self, env, root, journal, policy):
        self.env, self.root, self.journal, self.policy = env, Path(root), journal, policy
        self.count, self.fatal = 0, None
        self.deadline = time.monotonic() + journal.remaining()

    def __call__(self, env, action, *, config):
        self.journal.claim_action(self.root.name, self.count)
        folder = self.root / f"action-{self.count:02d}"
        folder.mkdir(exist_ok=False)
        self.count += 1
        try:
            outcome = PilotExecutor(folder, self.deadline)(env, action, config=config)
            handoff_id = outcome.evidence.get("image_grasp_handoff", {}).get(
                "before_observation_id"
            )
            if (
                self.policy.selection is None
                or self.policy.selection["observation_id"] != handoff_id
            ):
                raise RuntimeError("selection observation provenance mismatch")
            outcome.evidence["selection"] = self.policy.selection
            try:
                state_metrics(env)
            except RuntimeError as exc:
                # The primitive returned durable evidence. Do not discard it by
                # turning a post-execution safeguard into an empty collector exception.
                self.fatal = f"native state validation: {exc}"
                outcome.execution_valid = False
                outcome.terminated = True
                outcome.reason = self.fatal
                outcome.diagnostics["native_state_validation_error"] = str(exc)
                return outcome
            if getattr(env, "position_contact", False):
                contact = outcome.evidence.get("contact", {}).get("totals", {})
                if (
                    not contact
                    or not all(np.isfinite(v) for v in contact.values())
                    or contact.get("attached_friction_events", 1) != 0
                    or contact.get("positive_friction_work_j", 1) > 0
                ):
                    self.fatal = "native contact accounting regression"
            if campaign_stopping(outcome):
                self.fatal = outcome.reason
            if outcome.diagnostics.get("terminal_observation_valid") is False or (
                outcome.reason == "completed" and not outcome.execution_valid
            ):
                self.fatal = "invalid terminal artifact"
            for trace in outcome.evidence.get("phase_traces", []):
                for key in ("cloth_positions_m", "cloth_velocities_m_s"):
                    if key in trace and not np.isfinite(trace[key]).all():
                        self.fatal = "nonfinite native trace"
            if outcome.reason == "completed":
                audit = audit_tug_commands(
                    outcome.evidence.get("commands", []), target_distance_m=0.12
                )
                if not audit["passed"]:
                    self.fatal = "saved-command drag continuity regression"
            if outcome.diagnostics.get("settled") is not True:
                outcome.terminated = True
            return outcome
        except BaseException as exc:
            self.fatal = f"uncertain execution: {type(exc).__name__}: {exc}"
            raise


def saved_action_before_hash(root, index, support, execution_valid):
    """Recover pre-action evidence for failed execution without inventing a handoff."""
    before = support.get("evidence", {}).get("image_grasp_handoff", {}).get("before_state_hash")
    if before is None and not execution_valid:
        folder = Path(root) / f"action-{index:02d}"
        if (folder / "action-00-before.json").exists():
            before = read_hashed_bundle(folder, "action-00-before.json")["state_hash"]
    return before


def episode(
    env,
    journal,
    case,
    fixture,
    name,
    *,
    predict=None,
    epsilon=1.0,
    seed=0,
    coverage_stop=0.95,
    require_containment=False,
    reserve_seconds=None,
    max_actions=3,
):
    root = journal.root / "episodes" / name
    claim(
        root / "intent.json",
        {
            "case": case["id"],
            "family": case["family"],
            "seed": seed,
            "stage": journal.stage,
            "policy_epsilon": epsilon,
        },
    )
    policy = EpisodePolicy(predict, epsilon)
    executor = EpisodeExecutor(env, root, journal, policy)
    restore = env.restore
    calls = 0

    def checked_restore(state):
        nonlocal calls
        if calls:
            raise RuntimeError("restore between episode actions is forbidden")
        calls += 1
        # restore_exact must use the original native implementation, not this wrapper.
        env.restore = restore
        try:
            observation = restore_exact(env, state)
        finally:
            env.restore = checked_restore
        write(root, "restoration.json", {"passed": True, "state_hash": state["state_hash"]})
        return observation

    env.restore = checked_restore
    try:
        reserve = COLLECTION_RESERVE_SECONDS if journal.stage in ("exploration", "refresh") else 0
        if reserve_seconds is not None:
            reserve = reserve_seconds
        report = collect_episodes(
            env,
            root / "dataset",
            episodes=1,
            max_actions=max_actions,
            seed=seed,
            policy="image_learning_pilot",
            primitive="laydown",
            selector=policy,
            executor=executor,
            reset_state=fixture,
            max_wall_seconds=journal.remaining() - reserve,
            split=case["split"],
            reset_group_id=case["family"],
            episode_metadata={
                "initial_settled": True,
                "family": case["family"],
                "stage": journal.stage,
                "fixture_manifest_sha256": file_sha256(journal.root / "fixtures-manifest.json"),
            },
            settled_coverage_threshold=coverage_stop,
            success_requires_containment=require_containment,
        )
    finally:
        env.restore = restore
        cleanup = certify_release(env)
        write(root, "cleanup.json", cleanup)
    if not cleanup["passed"]:
        raise RuntimeError("episode native cleanup regression")
    with _reservation_deadline(time.monotonic() + journal.remaining()):
        verification = verify_dataset(root / "dataset")
        if not verification["valid"]:
            raise RuntimeError("episode artifact verification regression")
        rows = read_transitions(root / "dataset").to_pylist()
        details = []
        for i, row in enumerate(rows):
            support = read_support(root / "dataset", row["evidence_path"])
            native_chain_verified = True
            if i:
                for suffix in ("observation_id", "rgb_path", "support_path"):
                    if row[f"before_{suffix}"] != rows[i - 1][f"after_{suffix}"]:
                        raise RuntimeError("episode observation chain regression")
                state = read_support(root / "dataset", rows[i - 1]["final_state_path"])
                before_hash = saved_action_before_hash(root, i, support, row["execution_valid"])
                native_chain_verified = before_hash == state["state_hash"]
                if not native_chain_verified and (before_hash is not None or not executor.fatal):
                    raise RuntimeError("episode native state chain regression")
            diagnostics = support.get("diagnostics", {})
            concise_diagnostics = {
                k: diagnostics[k]
                for k in (
                    "settled",
                    "max_edge_ratio",
                    "tracking_error_m",
                    "final_rms_speed_m_s",
                    "final_table_containment",
                    "airborne_hold_seconds",
                    "airborne_hold_min_clearance_m",
                    "release_tcp_speed_m_s",
                    "terminal_observation_valid",
                    "frozen_laydown_version",
                    "leading_window_min_fraction",
                    "execution_mode",
                )
                if k in diagnostics
            }
            concise_diagnostics["native_chain_verified"] = native_chain_verified
            state_path = row.get("final_state_path")
            if row["execution_valid"] and not state_path:
                raise RuntimeError("missing native terminal state artifact")
            if state_path:
                native = read_support(root / "dataset", state_path)
                p = np.asarray(native["positions"])
                v = np.asarray(native["velocities"])
                if (
                    not np.isfinite(p).all()
                    or not np.isfinite(v).all()
                    or not np.asarray(native["free"]).all()
                ):
                    raise RuntimeError("native terminal finiteness/ownership regression")
                center = (p[:, :2].min(0) + p[:, :2].max(0)) / 2
                concise_diagnostics["final_center_error_m"] = float(
                    np.linalg.norm(center - env.config.table_center_xy_m)
                )
                concise_diagnostics["final_rms_speed_m_s"] = float(
                    np.sqrt(np.mean(np.sum(v**2, axis=1)))
                )
            details.append(
                {
                    k: row[k]
                    for k in (
                        "reason",
                        "execution_valid",
                        "coverage_before",
                        "coverage_after",
                        "coverage_gain",
                    )
                }
                | {
                    "diagnostics": concise_diagnostics,
                    "predicted_gain": support.get("evidence", {})
                    .get("selection", {})
                    .get("predicted_gain"),
                }
            )
        summary = report["episode_reports"][0]
        last = details[-1] if details else None
        good = bool(
            last
            and last["execution_valid"]
            and last["diagnostics"].get("settled") is True
            and summary.get("end_reason")
            not in ("no_valid_pair", "selection_error", "max_wall_seconds")
        )
        if not last and summary.get("lifecycle") == "initially_unfolded":
            good = True
        score = (last["coverage_after"] if last else summary["initial_coverage"]) if good else 0.0
        result = {
            "episode": name,
            "case": case["id"],
            "family": case["family"],
            "split": case["split"],
            "score": score,
            "failed": not good,
            "actions": details,
            "collection": report,
            "dataset_manifest_sha256": file_sha256(root / "dataset" / "manifest.json"),
            "fatal": executor.fatal or policy.fatal,
        }
        write(root, "result.json", result)
        print(f"{name}: {len(rows)} actions, score={score:.4f}, failed={not good}", flush=True)
        if executor.fatal or policy.fatal or report.get("reset_failures"):
            raise RuntimeError(executor.fatal or policy.fatal or "fixture restoration failure")
        return result


def collection_stage(env, journal, stage):
    records, states = fixtures(journal.root)
    predict = None
    epsilon, repeats = 1.0, 8
    if stage == "refresh":
        from cloth_fling.learning import predictor

        predict = predictor(journal.root / "checkpoints" / "initial.pt")
        epsilon, repeats = 0.5, 5
    results = []
    for case in records:
        if case["split"] != "train":
            continue
        for repetition in range(repeats):
            name = f"{stage}-{case['id']}-{repetition:02d}"
            results.append(
                episode(
                    env,
                    journal,
                    case,
                    states[case["id"]],
                    name,
                    predict=predict,
                    epsilon=epsilon,
                    seed=10000 + len(results),
                )
            )
    write(journal.root, f"{stage}-results.json", {"episodes": results})


def fit_stage(journal, stage):
    from cloth_fling.learning import fit

    _, _ = fixtures(journal.root)
    sources = ["exploration"] + (["refresh"] if stage == "refreshed" else [])
    datasets = []
    for source in sources:
        for result in read(journal.root, f"{source}-results.json")["episodes"]:
            folder = journal.root / "episodes" / result["episode"] / "dataset"
            if (
                file_sha256(folder / "manifest.json") != result["dataset_manifest_sha256"]
                or result["split"] != "train"
            ):
                raise ValueError("training dataset identity or split changed")
            datasets.append(folder)
    write(
        journal.root,
        f"{stage}-training.json",
        fit(
            datasets,
            journal.root / "checkpoints" / f"{stage}.pt",
            previous=journal.root / "checkpoints" / "initial.pt" if stage == "refreshed" else None,
            check_deadline=journal.check,
        ),
    )


def evaluation_stage(env, journal, split):
    from cloth_fling.learning import predictor

    records, states = fixtures(journal.root)
    if split == "test":
        selection = read(journal.root, "selection.json")
        if (
            file_sha256(journal.root / "checkpoints" / f"{selection['checkpoint']}.pt")
            != selection["weights_sha256"]
        ):
            raise ValueError("selected checkpoint changed before testing")
    candidates = (
        ["uniform", "initial", "refreshed"]
        if split == "validation"
        else ["uniform", read(journal.root, "selection.json")["checkpoint"]]
    )
    predictions = {
        name: predictor(journal.root / "checkpoints" / f"{name}.pt")
        for name in candidates
        if name != "uniform"
    }
    results = []
    for case in records:
        if case["split"] != split:
            continue
        for name in candidates:
            result = episode(
                env,
                journal,
                case,
                states[case["id"]],
                f"{split}-{case['id']}-{name}",
                predict=predictions.get(name),
                epsilon=1.0 if name == "uniform" else 0.0,
                seed=20000 + next(i for i, c in enumerate(records) if c["id"] == case["id"]),
            )
            result["policy"] = name
            results.append(result)
    write(journal.root, f"{split}-results.json", {"episodes": results})
    if split == "validation":
        values = {
            name: float(np.mean([r["score"] for r in results if r["policy"] == name]))
            for name in candidates
        }
        selected = "refreshed" if values["refreshed"] > values["initial"] else "initial"
        claim(
            journal.root / "selection.json",
            {
                "checkpoint": selected,
                "validation_scores": values,
                "weights_sha256": file_sha256(journal.root / "checkpoints" / f"{selected}.pt"),
            },
        )


def compare(results):
    groups = {}
    for result in results:
        case = groups.setdefault(result["family"], {}).setdefault(result["case"], {})
        key = "uniform" if result["policy"] == "uniform" else "learned"
        if key in case:
            raise ValueError("duplicate paired policy result")
        case[key] = result
    differences, failures = {}, {"uniform": 0, "learned": 0}
    for family, cases in groups.items():
        values = []
        for pair in cases.values():
            if set(pair) != {"uniform", "learned"}:
                raise ValueError("incomplete paired evaluation")
            values.append(pair["learned"]["score"] - pair["uniform"]["score"])
            for key in failures:
                failures[key] += int(pair[key]["failed"])
        differences[family] = float(np.mean(values))
    gain = float(np.mean(list(differences.values()))) if differences else None
    complete = len(groups) == 3 and all(len(cases) == 3 for cases in groups.values())
    return {
        "complete": complete,
        "family_gains": differences,
        "mean_paired_gain": gain,
        "failed_episodes": failures,
        "behavior_positive": bool(
            complete
            and gain >= 0.05
            and sum(x > 0 for x in differences.values()) >= 2
            and failures["learned"] <= failures["uniform"]
        ),
    }


def replay_stage(journal):
    from cloth_fling.replay import render_replay

    results = read(journal.root, "test-results.json")["episodes"]
    family = sorted({r["family"] for r in results})[0]
    videos = []
    for result in results:
        if result["family"] != family and not result["failed"]:
            continue
        source = journal.root / "episodes" / result["episode"] / "dataset"
        if not read_transitions(source).num_rows:
            continue
        journal.check()
        target = journal.root / "replays" / result["episode"]
        with _reservation_deadline(time.monotonic() + min(180, journal.remaining())):
            render_replay(source, target, fps=10, max_frames=2000)
        videos.append(
            {
                "relative_file": f"replays/{result['episode']}/combined.mp4",
                "sha256": file_sha256(target / "combined.mp4"),
            }
        )
    write(journal.root, "replays.json", videos)


def report(root, *, verify=True):
    """Offline reporting cannot reopen claims or turn a stopped pilot into success."""
    root = Path(root)
    started = time.monotonic()
    ledger = read(root, "ledger.json")
    intent_paths = sorted((root / "action-intents").glob("*.json"))
    intents = [read(root, str(p.relative_to(root))) for p in intent_paths]
    result = {
        "status": ledger["status"],
        "pipeline_complete": ledger["status"] == "complete" and verify,
        "verification_completed": verify,
        "charged_seconds": ledger["charged_seconds"],
        "claimed_actions": max(len(intents), len(ledger["actions"])),
        "action_limit": ACTION_LIMIT,
        "wall_limit_seconds": LIMIT_SECONDS,
        "frozen_config_sha256": CONFIG_SHA256,
        "material_qualification": False,
        "physical_pinch_grasp": False,
    }
    if ledger["status"] == "running_never_resume":
        result["status"] = "incomplete_uncertain_never_resume"
        result["uncertain_elapsed_seconds"] = max(
            0, time.time() - ledger["stages"][-1]["started_unix"]
        )
    integrity_errors = [] if intents == ledger["actions"] else ["action intent/ledger mismatch"]
    episode_results = []
    for stage in ledger["stages"]:
        if stage["status"] == "complete":
            for name, expected in stage["outputs"].items():
                if not (root / name).exists() or file_sha256(root / name) != expected:
                    integrity_errors.append(name)
    for path in sorted((root / "episodes").glob("*/result.json")):
        episode_result = json.loads(path.read_text())
        episode_results.append(episode_result)
        dataset = path.parent / "dataset"
        if (
            not (dataset / "manifest.json").exists()
            or file_sha256(dataset / "manifest.json") != episode_result["dataset_manifest_sha256"]
            or (verify and not verify_dataset(dataset)["valid"])
        ):
            integrity_errors.append(str(dataset.relative_to(root)))
    result["artifact_integrity_errors"] = integrity_errors
    actions = [a for e in episode_results for a in e["actions"]]
    gains = [a["coverage_gain"] for a in actions if a["coverage_gain"] is not None]
    strain = [
        a["diagnostics"]["max_edge_ratio"] for a in actions if "max_edge_ratio" in a["diagnostics"]
    ]
    result["observations"] = {
        "recorded_episodes": len(episode_results),
        "recorded_actions": len(actions),
        "failed_episodes": sum(e["failed"] for e in episode_results),
        "negative_coverage_gains": sum(g < 0 for g in gains),
        "maximum_material_edge_ratio": max(strain) if strain else None,
        "initial_and_final_coverage": [
            {
                "episode": e["episode"],
                "initial": e["collection"]["episode_reports"][0]["initial_coverage"],
                "final_score": e["score"],
                "failed": e["failed"],
            }
            for e in episode_results
        ],
    }
    if integrity_errors:
        result.update(status="incomplete_artifact_regression", pipeline_complete=False)
    if (root / "test-results.json").exists():
        result["comparison"] = compare(read(root, "test-results.json")["episodes"])
    result["behavior_positive"] = bool(
        result["pipeline_complete"] and result.get("comparison", {}).get("behavior_positive")
    )
    if (root / "replays.json").exists():
        result["replays"] = read(root, "replays.json")
    result["report_wall_seconds"] = time.monotonic() - started
    result["cumulative_execution_seconds"] = (
        ledger["charged_seconds"] + result["report_wall_seconds"]
    )
    if result["cumulative_execution_seconds"] > LIMIT_SECONDS:
        result.update(
            status="incomplete_budget_exhausted", pipeline_complete=False, behavior_positive=False
        )
    if not verify and result["status"] == "complete":
        result["status"] = "incomplete_reporting"
    write(root, "pilot-report.json", result)
    return result


def execute_stage(root, stage, *, backend_factory=None):
    root = Path(root)
    with (root / "execution.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(root)
        with (
            journal.operation(stage),
            reference_lease(root),
            _reservation_deadline(time.monotonic() + journal.remaining()),
        ):
            if stage in ("initial", "refreshed"):
                fit_stage(journal, stage)
                return
            if stage == "replays":
                replay_stage(journal)
                return
            if backend_factory is None:
                from cloth_fling.sim.reference import ReferenceBackend

                backend_factory = ReferenceBackend
            env = None
            try:
                with _reservation_deadline(time.monotonic() + min(180, journal.remaining())):
                    env = backend_factory(scene_config_from_dict(frozen_laydown_spec()["scene"]))
                if stage == "prepare":
                    prepare(env, journal)
                elif stage in ("exploration", "refresh"):
                    collection_stage(env, journal, stage)
                else:
                    evaluation_stage(env, journal, stage)
            finally:
                if env is not None:
                    certificate = certify_release(env)
                    write(root, f"{stage}-scene-cleanup.json", certificate)
                    env.close()
                    if not certificate["passed"]:
                        raise RuntimeError("scene cleanup failure")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "collect", "train", "evaluate", "report", "run")
    )
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    if args.command in ("run", "report") and args.stage is not None:
        parser.error("run/report do not accept --stage")
    stage_map = {
        "prepare": ("prepare",),
        "collect": ("exploration", "refresh"),
        "train": ("initial", "refreshed"),
        "evaluate": ("validation", "test", "replays"),
    }
    stages = STAGES if args.command == "run" else (args.stage or "prepare",)
    if args.command not in ("report", "run") and stages[0] not in stage_map[args.command]:
        parser.error("choose the matching --stage")
    if args.command in ("prepare", "run"):
        initialize(args.output)
    report_started = False
    try:
        if args.command != "report":
            for stage in stages:
                execute_stage(args.output, stage)
        remaining = LIMIT_SECONDS - read(args.output, "ledger.json")["charged_seconds"]
        report_started = True
        with _reservation_deadline(time.monotonic() + max(0.01, remaining)):
            result = report(args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["pipeline_complete"] or result["status"] == "between_stages" else 1
    except Exception as exc:
        if (args.output / "ledger.json").exists() and (
            report_started or read(args.output, "ledger.json")["status"] == "incomplete"
        ):
            write(args.output, "command-failure.json", {"error": f"{type(exc).__name__}: {exc}"})
            report(args.output, verify=False)
        print(f"learning pilot stopped: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
