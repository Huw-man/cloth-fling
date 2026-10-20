"""One fixture, two image-selected grasps, one frozen continuation; never resume."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from cloth_fling.benchmark_run import _reservation_deadline
from cloth_fling.collection import collect_episodes
from cloth_fling.contracts import scene_config_from_dict
from cloth_fling.data import read_support, read_transitions, verify_dataset
from cloth_fling.data.storage import atomic_write, file_sha256, json_bytes
from cloth_fling.frozen_laydown import execute_frozen_laydown, frozen_laydown_spec
from cloth_fling.provenance import runtime_provenance
from cloth_fling.reference_run import reference_lease, save_bundle
from cloth_fling.reference_tune import read_hashed_bundle
from cloth_fling.sim.mesh import visible_particle
from cloth_fling.spatial_actions import select_spatial_action

FIXTURE_SHA256 = "04a3b4083011bb2dbb5c749c57918c4090f6bbd5b302c01c267a3c3bfd2167f9"
FIXTURE_STATE_HASH = "e9b92de2d79a59b21f64029ddad6edf7b86f21184725e9d73dd0f355e4fd0445"
FIXTURE = "runs/friction-motion-demo-continued-20261019/fixture.json"
OUTPUT = "runs/image-grasp-laydown-pilot-centered-drag-v4-20261019"


def write(root, name, value):
    atomic_write(Path(root), name, json_bytes(value))


def resolve_grasp(env, action):
    """Read-only resolution, matching native attachment; ordering is not selection."""
    if env.attached_ids or not env._free().all():
        raise RuntimeError("grasp acquisition requires an entirely free cloth")
    positions = env.get_positions()
    ids = tuple(
        int(visible_particle(p, positions, env.faces, env.calibration, env.particle_diameter * 1.5))
        for p in action.pixels
    )
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("selected pixels must resolve to two distinct visible vertices")
    points = positions[list(ids)]
    xmin, xmax, ymin, ymax = env.config.table_bounds_m
    if not (
        np.isfinite(points).all()
        and env.config.min_grasp_separation_m
        <= np.linalg.norm(points[1] - points[0])
        <= env.config.max_grasp_separation_m
        and ((points[:, 0] >= xmin) & (points[:, 0] <= xmax)).all()
        and ((points[:, 1] >= ymin) & (points[:, 1] <= ymax)).all()
    ):
        raise ValueError("resolved grasp outside feasible workspace")
    order = sorted(range(2), key=lambda i: (points[i, 1], ids[i]))
    return {
        "requested_pixels": action.pixels,
        "action_encoding": getattr(action, "encoding", None),
        "resolved_native_ids": ids,
        "execution_permutation": order,
        "execution_native_ids": [ids[i] for i in order],
        "execution_pixels": [action.pixels[i] for i in order],
    }


def certify_release(env):
    """Cleanup only, never step/reset/restore, including partially failed acquisition."""
    try:
        held = bool(env.attached_ids)
        before = env.get_velocities().copy()
        env.release()
        release = env.release_evidence
        velocity_error = float(np.max(np.abs(env.get_velocities() - before)))
        certificate = {
            "released": not env.attached_ids,
            "native_free": bool(env._free().all()),
            "release": release,
            "cleanup_velocity_error_m_s": velocity_error,
        }
        certificate["passed"] = bool(
            certificate["released"]
            and certificate["native_free"]
            and (not held or release.get("passed"))
            and velocity_error <= 1e-4
        )
        return certificate
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


class PilotExecutor:
    def __init__(self, root, deadline):
        self.root, self.deadline = Path(root), deadline
        self.attempts = 0

    def __call__(self, env, action, *, config):
        index = self.attempts
        if index >= 2:
            raise RuntimeError("two-action limit exhausted")
        self.attempts += 1
        prefix = f"action-{index:02d}"
        # An exclusive immutable claim precedes resolution and any native mutation.
        with (self.root / f"{prefix}-intent.json").open("xb") as stream:
            stream.write(json_bytes({"action": action, "status": "claimed_never_retry"}))
            stream.flush()
            os.fsync(stream.fileno())
        certificate = None

        def cleanup(value):
            nonlocal certificate
            write(self.root, f"{prefix}-cleanup.json", value)
            certificate = value

        try:
            with _reservation_deadline(min(self.deadline, time.monotonic() + 180)):
                initial = env.snapshot()
                save_bundle(self.root, f"{prefix}-before", initial)
                handoff = resolve_grasp(env, action)
                handoff.update(
                    before_state_hash=initial["state_hash"],
                    before_observation_id=env.observe().observation_id,
                    incoming_simulated_time=env.simulated_time,
                )
                write(self.root, f"{prefix}-grasp.json", handoff)
                acquired = tuple(env.attach(tuple(handoff["execution_pixels"])))
                if acquired != tuple(handoff["execution_native_ids"]):
                    raise RuntimeError("native acquisition differs from pre-attachment resolution")
                if set(np.flatnonzero(~env._free())) != set(acquired):
                    raise RuntimeError("native ownership differs from the selected pair")
                save_bundle(self.root, f"{prefix}-attached", env.snapshot())
                result = execute_frozen_laydown(env, action, cleanup_callback=cleanup)
                if certificate is None or not certificate["passed"]:
                    raise RuntimeError("missing or failed native cleanup certification")
                result.evidence["image_grasp_handoff"] = handoff
                # This is an external historical artifact, not a dataset-local reference.
                reference = result.evidence["frozen_laydown"]["reference_evidence"]["outcome"]
                reference["source_relative_file"] = reference.pop("path")
                if (
                    result.reason != "completed"
                    or result.diagnostics.get("terminal_observation_valid") is False
                ):
                    result.execution_valid = False
                    result.terminated = True
                return result
        except BaseException as exc:
            write(self.root, f"{prefix}-failure.json", {"error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            if certificate is None:
                cleanup(certify_release(env))


def collect_pilot(env, root, fixture, deadline):
    """Reuse the durable collector; certify its sole restore before any selection."""
    restore = env.restore
    restores = 0

    def checked_restore(state):
        nonlocal restores
        if restores:
            raise RuntimeError("pilot cannot restore between actions")
        restores += 1
        observation = restore(state)
        actual = env.snapshot()
        passed = actual["state_hash"] == fixture["state_hash"] and all(
            np.array_equal(actual[key], fixture[key])
            for key in ("positions", "velocities", "free", "faces", "rest_positions")
        )
        write(
            root,
            "restoration.json",
            {
                "passed": passed,
                "expected_state_hash": fixture["state_hash"],
                "actual_state_hash": actual["state_hash"],
                "observation_id": observation.observation_id,
            },
        )
        if not passed:
            raise RuntimeError("exact reference fixture restoration regression")
        return observation

    env.restore = checked_restore
    try:
        return collect_episodes(
            env,
            Path(root) / "dataset",
            episodes=1,
            max_actions=2,
            seed=100,
            policy="uniform_valid_spatial",
            primitive="laydown",
            selector=select_spatial_action,
            executor=PilotExecutor(root, deadline),
            reset_state=fixture,
            max_wall_seconds=deadline - time.monotonic(),
            split="development",
            reset_group_id=fixture["state_hash"],
            episode_metadata={"initial_settled": True, "experiment": "two_action_integration"},
            settled_coverage_threshold=None,
        )
    finally:
        env.restore = restore


def audit_tug_commands(commands, *, target_distance_m=None):
    """Assess executed joins, including off-center shifts hidden by polynomial-only bounds."""
    lateral = vertical = speed = acceleration = distance = 0.0
    count = 0
    valid = bool(commands)
    anchor = None
    previous_velocity = np.zeros((2, 3))
    for index, command in enumerate(commands):
        if not index or not (command["phase"] == "tug" or commands[index - 1]["phase"] == "tug"):
            continue
        previous = commands[index - 1]
        a = np.asarray(previous["commanded_grasp_positions_m"], dtype=float)
        b = np.asarray(command["commanded_grasp_positions_m"], dtype=float)
        dt = command["simulated_seconds"] - previous["simulated_seconds"]
        if a.shape != (2, 3) or b.shape != (2, 3) or not np.isfinite([a, b]).all() or dt <= 0:
            valid = False
            continue
        if anchor is None:
            anchor = a.copy()
        velocity = (b - a) / dt
        lateral = max(lateral, float(np.abs(b[:, 1] - anchor[:, 1]).max()))
        vertical = max(vertical, float(np.abs(b[:, 2] - anchor[:, 2]).max()))
        speed = max(speed, float(np.linalg.norm(velocity, axis=1).max()))
        acceleration = max(
            acceleration, float(np.linalg.norm(velocity - previous_velocity, axis=1).max() / dt)
        )
        distance = max(distance, float((b[:, 0] - anchor[:, 0]).max()))
        valid = valid and bool((b[:, 0] >= a[:, 0] - 1e-9).all())
        previous_velocity = velocity
        count += int(command["phase"] == "tug")
    return {
        "passed": bool(
            valid
            and lateral <= 1e-9
            and vertical <= 1e-9
            and speed <= 0.8 + 1e-8
            and acceleration <= 2 + 1e-8
            and distance <= (target_distance_m if target_distance_m is not None else 0.02) + 1e-9
            and (target_distance_m is None or abs(distance - target_distance_m) <= 1e-8)
        ),
        "tug_controller_samples": count,
        "max_lateral_target_error_m": lateral,
        "max_vertical_target_error_m": vertical,
        "max_sampled_tcp_speed_m_s": speed,
        "max_sampled_tcp_acceleration_m_s2": acceleration,
        "forward_distance_m": distance,
    }


def assess(root):
    root = Path(root)
    verification = verify_dataset(root / "dataset")
    rows = read_transitions(root / "dataset").to_pylist()
    continuity = len(rows) == 2 and all(
        rows[1][f"before_{suffix}"] == rows[0][f"after_{suffix}"]
        for suffix in ("observation_id", "rgb_path", "support_path")
    )
    if continuity:
        first_state = read_support(root / "dataset", rows[0]["final_state_path"])
        second_evidence = read_support(root / "dataset", rows[1]["evidence_path"])["evidence"]
        continuity = (
            first_state["state_hash"] == second_evidence["image_grasp_handoff"]["before_state_hash"]
        )
    details = []
    for index, row in enumerate(rows):
        evidence = read_support(root / "dataset", row["evidence_path"])
        certificate = json.loads((root / f"action-{index:02d}-cleanup.json").read_text())
        phase_coverage = {
            trace["phase"]: trace.get("coverage")
            for trace in evidence["evidence"].get("phase_traces", [])
            if trace["phase"] in ("boundary_tug", "boundary_endpoint_hold")
        }
        drag_before = phase_coverage.get("boundary_tug")
        drag_after = phase_coverage.get("boundary_endpoint_hold")
        terminal_state = read_support(root / "dataset", row["final_state_path"])
        terminal_positions = np.asarray(terminal_state["positions"])
        footprint_center = (
            terminal_positions[:, :2].min(axis=0) + terminal_positions[:, :2].max(axis=0)
        ) / 2
        scene = json.loads(terminal_state["config_json"])
        workspace_center = np.asarray(scene["table_center_xy_m"])
        details.append(
            {
                "step": index,
                "execution_valid": row["execution_valid"],
                "reason": row["reason"],
                "coverage_before": row["coverage_before"],
                "coverage_after": row["coverage_after"],
                "coverage_gain": row["coverage_gain"],
                "diagnostics": evidence["diagnostics"],
                "cleanup_passed": certificate["passed"],
                "final_footprint_center_xy_m": footprint_center,
                "workspace_center_xy_m": workspace_center,
                "final_center_error_xy_m": footprint_center - workspace_center,
                "final_center_error_m": float(np.linalg.norm(footprint_center - workspace_center)),
                "placement_centering": evidence["evidence"].get("placement_centering"),
                "final_drag_coverage_before": drag_before,
                "final_drag_coverage_after": drag_after,
                "final_drag_coverage_gain": (
                    drag_after - drag_before
                    if drag_before is not None and drag_after is not None
                    else None
                ),
                "tug_continuity": audit_tug_commands(
                    evidence["evidence"].get("commands", []),
                    target_distance_m=(
                        evidence["evidence"].get("trajectory", {}).get("max_tug_m")
                        if evidence["evidence"].get("trajectory", {}).get("version")
                        in ("reference_diagonal_drag_v4", "reference_diagonal_centered_drag_v5")
                        else None
                    ),
                ),
            }
        )
    passed = bool(
        verification["valid"]
        and continuity
        and json.loads((root / "restoration.json").read_text())["passed"]
        and all(
            d["execution_valid"]
            and d["reason"] == "completed"
            and d["cleanup_passed"]
            and d["tug_continuity"]["passed"]
            for d in details
        )
    )
    return {
        "integration_passed": passed,
        "state_continuity": continuity,
        "dataset_verification": verification,
        "actions": details,
    }


def run_pilot(root, fixture_path, *, backend_factory=None, renderer=None, budget_predecessor=None):
    root, fixture_path = Path(root), Path(fixture_path)
    # Claim the entire experiment before constructing a scene; existing roots are terminal.
    root.mkdir(parents=True, exist_ok=False)
    parent_fd = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    started = time.monotonic()
    deadline = started + 3600
    report = {
        "status": "started_never_resume",
        "seed": 100,
        "action_limit": 2,
        "wall_limit_seconds": 3600,
        "qualification_claimed": False,
    }
    write(root, "pilot-report.json", report)
    env = None
    previous_charge = 0.0
    try:
        if budget_predecessor is not None:
            predecessor = json.loads(Path(budget_predecessor).read_text())
            previous_charge = predecessor.get(
                "cumulative_wall_seconds", predecessor["wall_seconds"]
            )
            if (
                predecessor["status"] not in ("complete", "incomplete")
                or not np.isfinite(previous_charge)
                or not 0 <= previous_charge < 3600
            ):
                raise ValueError("centering correction requires a terminal, accounted predecessor")
            deadline -= previous_charge
            report["budget_predecessor"] = {
                "source_relative_file": str(budget_predecessor),
                "sha256": file_sha256(budget_predecessor),
                "charged_wall_seconds": previous_charge,
            }
        if file_sha256(fixture_path) != FIXTURE_SHA256:
            raise ValueError("expected the exact archived reference fixture")
        fixture = read_hashed_bundle(fixture_path.parent, fixture_path.name)
        if fixture["state_hash"] != FIXTURE_STATE_HASH or not np.asarray(fixture["free"]).all():
            raise ValueError("reference fixture identity/ownership mismatch")
        save_bundle(root, "fixture", fixture)
        write(root, "runtime.json", runtime_provenance())
        if backend_factory is None:
            from cloth_fling.sim.reference import ReferenceBackend

            backend_factory = ReferenceBackend
        with reference_lease(root):
            try:
                with _reservation_deadline(min(deadline, time.monotonic() + 180)):
                    env = backend_factory(scene_config_from_dict(frozen_laydown_spec()["scene"]))
                report["collection"] = collect_pilot(env, root, fixture, deadline)
            finally:
                if env is not None:
                    write(root, "scene-cleanup.json", certify_release(env))
                    env.close()
            report.update(assess(root))
            scene_clean = json.loads((root / "scene-cleanup.json").read_text())["passed"]
            if not scene_clean:
                report["integration_passed"] = False
            write(root, "pilot-report.json", report)
            if not scene_clean or not report["dataset_verification"]["valid"]:
                raise RuntimeError("native cleanup or dataset artifact verification failed")
            if read_transitions(root / "dataset").num_rows:
                if renderer is None:
                    from cloth_fling.replay import render_replay

                    renderer = render_replay
                with _reservation_deadline(min(deadline, time.monotonic() + 180)):
                    replay = renderer(root / "dataset", root / "replay", fps=10, max_frames=800)
                report["replay"] = {"file": "replay/combined.mp4", "metadata": replay}
                if replay.get("truncated"):
                    raise RuntimeError("replay truncated")
        report["status"] = "complete" if report["integration_passed"] else "incomplete"
    except BaseException as exc:
        report.update(
            status="failed_or_uncertain",
            integration_passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["cumulative_wall_seconds"] = previous_charge + report["wall_seconds"]
        write(root, "pilot-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    parser.add_argument("--fixture", type=Path, default=Path(FIXTURE))
    parser.add_argument(
        "--budget-predecessor",
        type=Path,
        default=Path("runs/image-grasp-laydown-pilot-drag-v3-20261019/pilot-report.json"),
    )
    args = parser.parse_args()
    report = run_pilot(args.output, args.fixture, budget_predecessor=args.budget_predecessor)
    print(
        json.dumps({key: report[key] for key in ("status", "integration_passed", "wall_seconds")})
    )
    return 0 if report["integration_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
