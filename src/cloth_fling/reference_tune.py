"""One durable, non-resumable development campaign per archived reference fixture."""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cloth_fling.contracts import GraspAction, ReferenceSceneConfig
from cloth_fling.data import read_support
from cloth_fling.data.storage import atomic_write, file_sha256
from cloth_fling.reference_leading_motion import (
    acceptance,
    campaign_stopping,
    execute_leading_action,
)
from cloth_fling.reference_run import _write, reference_lease, save_bundle
from cloth_fling.reference_trajectory import candidate_grid, placement_schedule


class TuneBudget:
    def __init__(self, root):
        self.root, self.started = Path(root), time.monotonic()
        if (self.root / "tune-budget.json").exists():
            raise RuntimeError("terminal or uncertain campaign cannot resume")
        self.document = {
            "limit_wall_seconds": 7200,
            "limits": {"development": 20, "confirmation": 3},
            "operations": [],
        }
        self.save()

    def save(self):
        _write(self.root, "tune-budget.json", self.document)

    @contextmanager
    def reserve(self, category, allowance, **metadata):
        from cloth_fling.benchmark_run import _reservation_deadline

        if category not in ("preparation", "contact", "development", "confirmation"):
            raise ValueError("unknown tuning operation")
        ops = self.document["operations"]
        if (
            category in self.document["limits"]
            and sum(o["category"] == category for o in ops) >= self.document["limits"][category]
        ):
            raise RuntimeError("tuning attempt budget exhausted")
        charged = sum(o.get("charged_seconds", o["allowance_seconds"]) for o in ops)
        remaining = 7200 - max(charged, time.monotonic() - self.started)
        if not np.isfinite(allowance) or not 0 < allowance <= remaining:
            raise RuntimeError("tuning wall budget exhausted")
        entry = {
            "category": category,
            "allowance_seconds": allowance,
            "status": "reserved",
            **metadata,
        }
        ops.append(entry)
        self.save()  # intent reaches stable storage before scene mutation
        started = time.monotonic()
        try:
            with _reservation_deadline(min(self.started + 7200, started + allowance)):
                yield entry
        except BaseException:
            entry.update(status="failed_or_uncertain", charged_seconds=allowance)
            raise
        else:
            entry.update(status="finished", charged_seconds=time.monotonic() - started)
        finally:
            self.save()


def read_hashed_bundle(root, reference):
    path = root / reference
    document = json.loads(path.read_text())
    arrays = document.get("arrays")
    if arrays and file_sha256(root / arrays["path"]) != arrays["sha256"]:
        raise ValueError("archived bundle array hash mismatch")
    return read_support(root, reference)


def restore_exact(env, state, expected_rgb, expected_id):
    restored = env.restore(state)
    if (
        env.snapshot()["state_hash"] != state["state_hash"]
        or restored.observation_id != expected_id
        or not np.array_equal(restored.rgb, expected_rgb)
    ):
        raise RuntimeError("exact fixture restoration/rendering regression")
    return restored


def _archive_sources(root):
    result = {}
    for source in sorted(Path(__file__).parent.rglob("*.py")):
        ref = "source/" + source.relative_to(Path(__file__).parent).as_posix()
        atomic_write(root, ref, source.read_bytes())
        result[ref] = file_sha256(root / ref)
    return result


def run_reference_tuning(output_dir, *, fixture_dir, backend="gpu"):
    root, fixture_dir = Path(output_dir).resolve(), Path(fixture_dir).resolve()
    # No scene, lease, new directory or budget allocation on repeat invocation.
    if root.exists():
        if (root / "tune-report.json").exists():
            report = json.loads((root / "tune-report.json").read_text())
            if report["status"] == "started":
                return {**report, "status": "uncertain_incomplete", "stage_passed": False}
            return report
        raise RuntimeError("existing output is terminal or uncertain; cannot resume")
    prior_report = json.loads((fixture_dir / "reference-report.json").read_text())
    if (
        prior_report.get("status") != "placement_limitation"
        or not prior_report.get("restoration_passed")
        or not prior_report.get("contact", {}).get("passed")
    ):
        raise ValueError("requires the archived failed reference campaign")
    fixture_ref = prior_report["initial_state"]
    if file_sha256(fixture_dir / fixture_ref["path"]) != fixture_ref["sha256"]:
        raise ValueError("fixture metadata hash mismatch")
    state = read_hashed_bundle(fixture_dir, fixture_ref["path"])
    outcome_ref = prior_report["single_action"]
    if file_sha256(fixture_dir / outcome_ref["path"]) != outcome_ref["sha256"]:
        raise ValueError("historical action metadata hash mismatch")
    prior = read_hashed_bundle(fixture_dir, outcome_ref["path"])
    expected_ids = prior["evidence"]["commands"][0]["attachment_ids"]
    action = GraspAction(**prior["action"])
    if (
        len(expected_ids) != 2
        or state["schema_version"] != 2
        or json.loads(state["config_json"])
        != json.loads(json.dumps(ReferenceSceneConfig().to_dict()))
    ):
        raise ValueError("archived fixture/vertex identity mismatch")
    candidates = []
    for index, candidate in enumerate(candidate_grid()):
        try:
            _, schedule = placement_schedule(candidate)
            candidates.append(
                {
                    "index": index,
                    "trajectory": asdict(candidate),
                    "schedule": schedule,
                    "offline_accepted": True,
                }
            )
        except ValueError as exc:
            candidates.append(
                {
                    "index": index,
                    "trajectory": asdict(candidate),
                    "offline_accepted": False,
                    "reason": str(exc),
                }
            )
    # Keep historical campaign directories untouched. Recognize the original claim
    # location too, so migrating its record can never reopen an execution window.
    claim_directory = fixture_dir.parent / "reference-leading-v2-claims"
    claim_directory.mkdir(exist_ok=True)
    key = hashlib.sha256(state["state_hash"].encode()).hexdigest()
    claim = claim_directory / f"{key}.json"
    if claim.exists() or (fixture_dir / "reference-leading-v2-claim.json").exists():
        raise RuntimeError("this fixture already has a terminal or uncertain tuning campaign")
    root.mkdir(parents=True, exist_ok=False)
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            "this fixture already has a terminal or uncertain tuning campaign"
        ) from exc
    with os.fdopen(fd, "w") as stream:
        json.dump(
            {
                "output": str(root),
                "fixture_hash": state["state_hash"],
                "version": "reference_leading_v2",
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(claim_directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    budget = TuneBudget(root)
    report = {
        "schema": "cloth_fling.reference_tune.v2",
        "status": "started",
        "stage_passed": False,
        "fixture_state_hash": state["state_hash"],
        "favorable_vertex_ids": expected_ids,
        "fixture_ancestry": {
            "source": str(fixture_dir),
            "reference_report_sha256": file_sha256(fixture_dir / "reference-report.json"),
        },
        "candidates": candidates,
        "attempts": [],
        "frozen_candidate": None,
        "learning": "not_evaluated",
        "varied_grasps": "not_evaluated",
        "material_qualification": "not_evaluated",
    }
    _write(root, "tune-report.json", report)
    env = None
    try:
        report["source_files"] = _archive_sources(root)
        save_bundle(root, "fixture", state)
        _write(
            root,
            "configuration.json",
            {
                "scene": ReferenceSceneConfig().to_dict(),
                "candidates": candidates,
                "favorable_action": asdict(action),
                "expected_ids": expected_ids,
            },
        )
        with reference_lease(root):
            try:
                from cloth_fling.sim.appearance import AppearanceConfig
                from cloth_fling.sim.reference import ReferenceBackend

                with budget.reserve("preparation", 600):
                    env = ReferenceBackend(
                        ReferenceSceneConfig(), backend=backend, appearance=AppearanceConfig()
                    )
                    restore_exact(
                        env, state, prior["before"]["rgb"], prior["before"]["observation_id"]
                    )
                    report["restoration_passed"] = True
                with budget.reserve("contact", 180):
                    p = env.rest_positions.copy()
                    p[:, 2] = 0.305
                    p[:, :2] += env.config.cloth_center_m[:2]
                    env.cloth.set_particles_pos(p)
                    v = np.zeros_like(p)
                    v[:, 0] = 0.1
                    env.cloth.set_particles_vel(v)
                    env.take_contact_accounting(reset=True)
                    env.step(100)
                    speed = float(np.linalg.norm(env.get_velocities()[:, :2], axis=1).mean())
                    accounting = env.take_contact_accounting()
                    report["contact"] = {
                        "final_mean_speed_m_s": speed,
                        "accounting": accounting,
                        "passed": bool(speed < 0.1 and accounting["corrected_particles"] > 0),
                    }
                    save_bundle(root, "contact-terminal", env.snapshot())
                    if not report["contact"]["passed"]:
                        raise RuntimeError("contact regression")

                def attempt(index, category, candidate):
                    from cloth_fling.reference_trajectory import LeadingTrajectory

                    number = len(report["attempts"])
                    with budget.reserve(category, 180, attempt=number, candidate=index) as receipt:
                        restore_exact(
                            env, state, prior["before"]["rgb"], prior["before"]["observation_id"]
                        )
                        env.take_contact_accounting(reset=True)
                        value = execute_leading_action(
                            env,
                            action,
                            LeadingTrajectory(**candidate["trajectory"]),
                            expected_ids=expected_ids,
                            cleanup_callback=lambda c: _write(
                                root, f"cleanup/attempt-{number:02d}.json", c
                            ),
                        )
                        reference = save_bundle(root, f"attempt-{number:02d}", value)
                        gates = acceptance(value)
                        record = {
                            "attempt": number,
                            "category": category,
                            "candidate": index,
                            "outcome": reference,
                            "reason": value.reason,
                            "gates": gates,
                            "passed": all(gates.values()),
                            "diagnostics": {
                                k: v for k, v in value.diagnostics.items() if k != "tauten"
                            },
                            "coverage_after": value.coverage_after,
                            "simulated_seconds": value.simulated_seconds,
                        }
                        report["attempts"].append(record)
                        receipt["outcome"] = reference
                        _write(root, "tune-report.json", report)
                        return record, campaign_stopping(value)

                for candidate in candidates:
                    if not candidate["offline_accepted"]:
                        continue
                    record, fatal = attempt(candidate["index"], "development", candidate)
                    if fatal:
                        report["status"] = "campaign_stopping_failure"
                        break
                    if record["passed"]:
                        report["frozen_candidate"] = candidate
                        _write(root, "frozen-trajectory.json", candidate)
                        _write(root, "tune-report.json", report)
                        confirmations_passed = True
                        for _ in range(3):
                            confirmation, fatal = attempt(
                                candidate["index"], "confirmation", candidate
                            )
                            confirmations_passed &= confirmation["passed"]
                            if fatal:
                                report["status"] = "campaign_stopping_failure"
                                break
                        else:
                            report.update(
                                status="complete"
                                if confirmations_passed
                                else "confirmation_incomplete",
                                stage_passed=confirmations_passed,
                            )
                        break
                else:
                    report["status"] = "placement_limitation"
            finally:
                if env is not None:
                    try:
                        env.release()
                        cleanup = {
                            "released": not env.attached_ids,
                            "native_free": bool(env._free().all()),
                        }
                        _write(root, "final-cleanup.json", cleanup)
                        if not all(cleanup.values()):
                            raise RuntimeError("final native cleanup regression")
                    finally:
                        env.close()
    except BaseException as exc:
        report.update(status="stopped", stage_passed=False, error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            raise
    finally:
        from cloth_fling.provenance import runtime_provenance

        report["runtime_provenance"] = runtime_provenance()
        report["elapsed_wall_seconds"] = time.monotonic() - budget.started
        _write(root, "tune-report.json", report)
    return report
