"""Bounded reference qualification. Terminal attempts cannot be resumed or extended."""

import fcntl
import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cloth_fling.contracts import GraspAction, ReferenceSceneConfig
from cloth_fling.data.storage import atomic_write, file_sha256, json_bytes, write_bundle
from cloth_fling.reference_motion import ReferenceMotionConfig, execute_reference_action


def _write(root, name, value):
    atomic_write(root, name, json_bytes(value))


def save_bundle(root, name, value):
    def put(path, data):
        atomic_write(root, path, data)
        return {"path": path, "sha256": file_sha256(root / path)}

    return write_bundle(put, value, name)


@contextmanager
def reference_lease(root):
    with Path("/tmp/cloth-fling-rtx5080.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("exclusive simulator lease is already held") from exc
        _write(root, "gpu-lease.json", {"owner": "reference-coordinator", "active": True})
        try:
            yield
        finally:
            _write(root, "gpu-lease.json", {"owner": "reference-coordinator", "active": False})


class ReferenceBudget:
    def __init__(self, root, previous=None):
        self.root = Path(root)
        self.started = time.monotonic()
        self.document = {
            "limit_wall_seconds": 3600,
            "limits": {"diagnostic": 6, "continuity": 6, "episode": 2},
            "operations": [],
            "episodes": 0,
        }
        if previous is not None:
            previous = Path(previous).resolve()
            prior = json.loads((previous / "reference-budget.json").read_text())
            prior_report = json.loads((previous / "reference-report.json").read_text())
            if prior_report["status"] == "started" or any(
                op["status"] == "reserved" for op in prior["operations"]
            ):
                raise RuntimeError("unreconciled prior reference operation")
            self.document["operations"] = prior["operations"]
            self.document["episodes"] = prior["episodes"]
            self.document["predecessor"] = {
                "directory": str(previous),
                "budget_sha256": file_sha256(previous / "reference-budget.json"),
            }
        self._save()

    def _save(self):
        _write(self.root, "reference-budget.json", self.document)

    @contextmanager
    def reserve(self, category, allowance):
        from cloth_fling.benchmark_run import _reservation_deadline

        if category not in ("preparation", "contact", "diagnostic", "continuity"):
            raise ValueError("unknown reference budget category")
        if (
            category in self.document["limits"]
            and sum(op["category"] == category for op in self.document["operations"])
            >= self.document["limits"][category]
        ):
            raise RuntimeError("reference attempt budget exhausted")
        charged = sum(
            op.get("charged_seconds", op["allowance_seconds"]) for op in self.document["operations"]
        )
        remaining = 3600 - max(charged, time.monotonic() - self.started)
        if not np.isfinite(allowance) or not 0 < allowance <= remaining:
            raise RuntimeError("reference wall budget exhausted")
        entry = {"category": category, "allowance_seconds": allowance, "status": "reserved"}
        self.document["operations"].append(entry)
        self._save()
        started = time.monotonic()
        try:
            with _reservation_deadline(min(self.started + 3600, started + allowance)):
                yield
        except BaseException:
            entry.update(status="failed_or_uncertain", charged_seconds=allowance)
            raise
        else:
            entry.update(status="finished", charged_seconds=time.monotonic() - started)
        finally:
            self._save()

    def claim_episode(self):
        if self.document["episodes"] >= 2:
            raise RuntimeError("reference episode budget exhausted")
        self.document["episodes"] += 1
        self._save()


def _project(points, calibration):
    inverse = np.linalg.inv(np.asarray(calibration["camera_to_world"]))
    camera = np.c_[points, np.ones(len(points))] @ inverse.T
    uvz = camera[:, :3] @ np.asarray(calibration["intrinsics"]).T
    return uvz[:, :2] / uvz[:, 2:] - calibration["pixel_center_offset"]


def favorable_action(env):
    """Development-only pair just inside the two corners of the negative-X short edge."""
    rest = env.rest_positions
    targets = np.array(
        [
            [rest[:, 0].min() + 0.015, rest[:, 1].min() + 0.015],
            [rest[:, 0].min() + 0.015, rest[:, 1].max() - 0.015],
        ]
    )
    ids = [int(np.argmin(np.linalg.norm(rest[:, :2] - point, axis=1))) for point in targets]
    pixels = _project(env.get_positions()[ids], env.calibration)
    return GraspAction(tuple(map(tuple, pixels)), policy="fixed_material_corners")


def placement_passed(outcome):
    d = outcome.diagnostics
    return bool(
        outcome.execution_valid
        and d.get("settled")
        and d.get("airborne_hold_seconds", 0) >= 3 - 1e-8
        and d.get("airborne_hold_min_clearance_m", -1) > 0
        and d.get("tracking_error_m", 1) <= 1e-6
        and d.get("release_tcp_speed_m_s", 1) <= 0.01
        and d.get("final_table_containment") == 1.0
        and outcome.evidence["cleanup"]["passed"]
        and outcome.evidence["cleanup"]["release"].get("passed", False)
    )


def run_reference_qualification(output_dir, *, backend="gpu", previous=None):
    from cloth_fling.provenance import runtime_provenance

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    budget = ReferenceBudget(root, previous=previous)
    report = {
        "profile": "reference_laydown_v1",
        "stage_passed": False,
        "motion": "controlled_laydown",
        "learning": "not_evaluated",
        "dynamic_fling": "not_evaluated",
        "continuity": [],
        "status": "started",
    }
    config, motion = ReferenceSceneConfig(), ReferenceMotionConfig()
    _write(
        root,
        "configuration.json",
        {
            "scene": config.to_dict(),
            "motion": asdict(motion),
            "ancestry": {"split": "development", "reset_group_id": "reference-authored-v1"},
        },
    )
    if previous is not None:
        if any(
            op["category"] in ("diagnostic", "continuity") for op in budget.document["operations"]
        ):
            raise ValueError(
                "predecessor has terminal motion evidence; dependent execution stopped"
            )
        previous_config = json.loads((Path(previous) / "configuration.json").read_text())
        if previous_config["scene"] != json.loads(json.dumps(config.to_dict())):
            raise ValueError("predecessor reference physics differ")
    source_root = Path(__file__).parent
    report["source_files"] = {}
    for source in sorted(source_root.rglob("*.py")):
        reference = "source/" + source.relative_to(source_root).as_posix()
        atomic_write(root, reference, source.read_bytes())
        report["source_files"][reference] = file_sha256(root / reference)
    env = None
    try:
        with reference_lease(root):
            try:
                from cloth_fling.sim.appearance import AppearanceConfig
                from cloth_fling.sim.reference import ReferenceBackend

                with budget.reserve("preparation", 600):
                    env = ReferenceBackend(config, backend=backend, appearance=AppearanceConfig())
                    observation = env.reset(kind="authored")
                    state = env.snapshot()
                    report["initial_state"] = save_bundle(root, "fixture", state)
                    restored = env.restore(state)
                    if (
                        restored.observation_id != observation.observation_id
                        or not np.array_equal(restored.rgb, observation.rgb)
                        or env.snapshot()["state_hash"] != state["state_hash"]
                    ):
                        raise RuntimeError("reference restoration/rendering mismatch")
                    report["geometry"] = {
                        "particles": len(env.rest_positions),
                        "triangles": len(env.faces),
                        "native_area_m2": env.native_rest_area_m2,
                        "authored_area_m2": 0.24,
                    }
                    report["restoration_passed"] = True
                # Native resting-slide coupon uses free-particle initialization, never control.
                with budget.reserve("contact", 180):
                    p = env.rest_positions.copy()
                    p[:, 2] = config.table_surface_z_m + config.mesh_spacing_m / 2
                    p[:, :2] += config.cloth_center_m[:2]
                    env.cloth.set_particles_pos(p)
                    velocity = np.zeros_like(p)
                    velocity[:, 0] = 0.1
                    env.cloth.set_particles_vel(velocity)
                    env.take_contact_accounting(reset=True)
                    env.step(100)
                    after_speed = float(np.linalg.norm(env.get_velocities()[:, :2], axis=1).mean())
                    accounting = env.take_contact_accounting()
                    report["contact"] = {
                        "initial_mean_speed_m_s": 0.1,
                        "final_mean_speed_m_s": after_speed,
                        "accounting": accounting,
                        "passed": bool(after_speed < 0.1 and accounting["corrected_particles"] > 0),
                    }
                    save_bundle(root, "contact-terminal", env.snapshot())
                    if not report["contact"]["passed"]:
                        raise RuntimeError("reference contact coupon failed")
                    env.restore(state)
                counter = 0

                def execute(current_env, action, *, config=motion, trace_callback=None):
                    nonlocal counter
                    counter += 1
                    category = "diagnostic" if counter == 1 else "continuity"
                    with budget.reserve(category, 180):
                        return execute_reference_action(
                            current_env,
                            action,
                            config=config,
                            trace_callback=trace_callback,
                            cleanup_callback=lambda value: _write(
                                root, f"cleanup/action-{counter}.json", value
                            ),
                        )

                outcome = execute(env, favorable_action(env))
                report["single_action"] = save_bundle(root, "single-action", outcome)
                report["placement_passed"] = placement_passed(outcome)
                _write(root, "reference-report.json", report)
                if not report["placement_passed"]:
                    report["status"] = "placement_limitation"
                    return report
                # Fresh settled authored fixtures use the same ancestry; no generalization claim.
                from cloth_fling.collection import collect_episodes
                from cloth_fling.data import verify_dataset
                from cloth_fling.spatial_actions import select_spatial_action

                with budget.reserve("preparation", 180):
                    env.reset(kind="authored")
                    stable = 0
                    for _ in range(250):
                        env.step(20)
                        speed = float(np.sqrt(np.mean(np.sum(env.get_velocities() ** 2, axis=1))))
                        stable = stable + 1 if speed <= 0.01 else 0
                        if stable >= 5:
                            break
                    if stable < 5:
                        raise RuntimeError("reference episode fixture did not settle")
                    settled_state = env.snapshot()
                for episode in range(2):
                    budget.claim_episode()
                    with budget.reserve("preparation", 30):
                        env.restore(settled_state)
                    result = collect_episodes(
                        env,
                        root / f"continuity-{episode}",
                        episodes=1,
                        max_actions=3,
                        seed=episode,
                        policy="uniform_valid_spatial",
                        primitive="laydown",
                        selector=select_spatial_action,
                        executor=execute,
                        primitive_config=motion,
                        max_wall_seconds=min(
                            600, max(1, 3600 - (time.monotonic() - budget.started))
                        ),
                        reset_state=settled_state,
                        reset_kind="authored",
                        split="development",
                        reset_group_id="reference-authored-v1",
                        cloth_id="reference-towel-v1",
                        episode_metadata={"initial_settled": True, "qualification_only": True},
                    )
                    verified = verify_dataset(root / f"continuity-{episode}")
                    report["continuity"].append({"result": result, "verification": verified})
                    if result["valid"] < 2 or result["invalid"] or not verified.get("valid", False):
                        report["status"] = "continuity_limitation"
                        return report
                report.update(status="complete", stage_passed=True)
            finally:
                if env is not None:
                    try:
                        env.release()
                        final_cleanup = {
                            "native_free": bool(env._free().all()),
                            "released": not env.attached_ids,
                        }
                        _write(root, "final-cleanup.json", final_cleanup)
                        if not all(final_cleanup.values()):
                            raise RuntimeError("final native cleanup verification failed")
                    finally:
                        env.close()
    except Exception as exc:
        report.update(status="stopped", stage_passed=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        report["runtime_provenance"] = runtime_provenance()
        _write(root, "reference-report.json", report)
    return report
