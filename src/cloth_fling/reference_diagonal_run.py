"""One durable two-stage campaign; contact qualification gates every motion attempt."""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cloth_fling.contracts import GraspAction
from cloth_fling.data.storage import file_sha256, json_bytes
from cloth_fling.reference_diagonal_motion import (
    acceptance,
    campaign_stopping,
    diagonal_grid,
    diagonal_schedule,
    execute_diagonal_action,
)
from cloth_fling.reference_run import _write, reference_lease, save_bundle
from cloth_fling.reference_surface import (
    contact_cases,
    diagonal_scene,
    import_parent_fixture,
    run_contact_case,
)
from cloth_fling.reference_tune import _archive_sources, read_hashed_bundle, restore_exact


class DiagonalBudget:
    def __init__(self, root, *, create=False):
        self.root = Path(root)
        self.started = time.monotonic()
        path = self.root / "diagonal-budget.json"
        if create:
            if path.exists():
                raise RuntimeError("existing diagonal campaign cannot restart")
            self.document = {
                "limit_wall_seconds": 7200,
                "limits": {"contact": 16, "development": 12, "confirmation": 3},
                "operations": [],
            }
            self.save()
        else:
            self.document = json.loads(path.read_text())
            if any(o["status"] != "finished" for o in self.document["operations"]):
                raise RuntimeError("uncertain or failed reservation cannot resume")
        self.prior_charged = self.charged()

    def charged(self):
        return sum(
            o.get("charged_seconds", o["allowance_seconds"]) for o in self.document["operations"]
        )

    def save(self):
        _write(self.root, "diagonal-budget.json", self.document)

    @contextmanager
    def reserve(self, category, allowance, **metadata):
        from cloth_fling.benchmark_run import _reservation_deadline

        if category not in ("preparation", "contact", "development", "confirmation", "render"):
            raise ValueError("unknown diagonal operation")
        ops = self.document["operations"]
        if (
            category in self.document["limits"]
            and sum(o["category"] == category for o in ops) >= self.document["limits"][category]
        ):
            raise RuntimeError("diagonal attempt budget exhausted")
        elapsed = self.prior_charged + time.monotonic() - self.started
        remaining = 7200 - max(self.charged(), elapsed)
        if not np.isfinite(allowance) or not 0 < allowance <= remaining:
            raise RuntimeError("diagonal wall budget exhausted")
        entry = {
            "category": category,
            "allowance_seconds": allowance,
            "status": "reserved",
            **metadata,
        }
        ops.append(entry)
        self.save()
        start = time.monotonic()
        try:
            with _reservation_deadline(start + allowance):
                yield entry
        except BaseException:
            entry.update(status="failed_or_uncertain", charged_seconds=allowance)
            raise
        else:
            entry.update(status="finished", charged_seconds=time.monotonic() - start)
        finally:
            self.save()


def terminal_report(root, name):
    root = Path(root)
    if (root / name).exists():
        report = json.loads((root / name).read_text())
        if report["status"] == "started":
            return {**report, "status": "uncertain_incomplete", "stage_passed": False}
        return report
    if root.exists():
        return {
            "status": "uncertain_incomplete",
            "stage_passed": False,
            "error": "existing directory has no terminal report",
        }
    return None


def _exclusive_claim(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _cleanup(env, root, name):
    try:
        env.release()
        certificate = {"released": not env.attached_ids, "native_free": bool(env._free().all())}
        certificate["passed"] = all(certificate.values())
        _write(root, name, certificate)
        if not certificate["passed"]:
            raise RuntimeError("native cleanup regression")
    finally:
        env.close()


def _finish(root, name, report, budget):
    from cloth_fling.provenance import runtime_provenance

    report["runtime_provenance"] = runtime_provenance()
    report["charged_simulator_wall_seconds"] = budget.charged()
    _write(root, name, report)


def run_contact_qualification(output_dir, *, fixture_dir, backend="gpu"):
    root, source = Path(output_dir).resolve(), Path(fixture_dir).resolve()
    previous = terminal_report(root, "contact-report.json")
    if previous is not None:
        return previous
    prior = json.loads((source / "reference-report.json").read_text())
    fixture_ref = prior["initial_state"]
    action_ref = prior["single_action"]
    for ref in (fixture_ref, action_ref):
        if file_sha256(source / ref["path"]) != ref["sha256"]:
            raise ValueError("parent archive hash mismatch")
    parent = read_hashed_bundle(source, fixture_ref["path"])
    old = read_hashed_bundle(source, action_ref["path"])
    key = hashlib.sha256(parent["state_hash"].encode()).hexdigest()
    claim = source.parent / "reference-diagonal-v3-claims" / f"{key}.json"
    _exclusive_claim(
        claim,
        {
            "output": str(root),
            "parent_state_hash": parent["state_hash"],
            "version": "reference_diagonal_v3",
        },
    )
    root.mkdir(parents=True, exist_ok=False)
    budget = DiagonalBudget(root, create=True)
    cases = contact_cases()
    report = {
        "schema": "cloth_fling.reference_contact.v3",
        "status": "started",
        "stage_passed": False,
        "cases": [],
        "parent_state_hash": parent["state_hash"],
        "parent_report_sha256": file_sha256(source / "reference-report.json"),
        "parent_directory": str(source),
        "favorable_vertex_ids": old["evidence"]["commands"][0]["attachment_ids"],
        "motion_evaluated": False,
    }
    _write(root, "contact-report.json", report)
    try:
        report["source_files"] = _archive_sources(root)
        save_bundle(root, "parent-fixture", parent)
        config = diagonal_scene()
        _write(
            root,
            "configuration.json",
            {
                "scene": config.to_dict(),
                "contact_cases": cases,
                "motion_grid": [
                    {"index": i, "trajectory": asdict(c), "schedule": diagonal_schedule(c)[1]}
                    for i, c in enumerate(diagonal_grid())
                ],
                "favorable_action": old["action"],
                "expected_ids": report["favorable_vertex_ids"],
                "limits": budget.document["limits"],
            },
        )
        report["configuration_sha256"] = file_sha256(root / "configuration.json")
        with reference_lease(root):
            from cloth_fling.sim.appearance import AppearanceConfig
            from cloth_fling.sim.reference import ReferenceBackend

            with budget.reserve("preparation", 600):
                env = ReferenceBackend(config, backend=backend, appearance=AppearanceConfig())
                try:
                    state = import_parent_fixture(env, parent)
                    observation = env.observe()
                    restore_exact(env, state, observation.rgb, observation.observation_id)
                    # Also verify restored native depth, independent of the RGB/ID check.
                    if not np.array_equal(env.observe().depth, observation.depth):
                        raise RuntimeError("fixture depth restoration regression")
                    report["fixture"] = save_bundle(root, "fixture", state)
                    report["fixture_observation"] = save_bundle(
                        root, "fixture-observation", observation
                    )
                    report["fixture_state_hash"] = state["state_hash"]
                    report["restoration_passed"] = True
                finally:
                    _cleanup(env, root, "cleanup/preparation.json")
            for index, case in enumerate(cases):
                with budget.reserve("contact", 60, case=case["name"], index=index) as receipt:

                    def failure_artifact(value):
                        ref = save_bundle(root, f"contact-{index:02d}-failure", value)
                        report.setdefault("failure_artifacts", []).append(
                            {"case": case["name"], "outcome": ref}
                        )
                        _write(root, "contact-report.json", report)

                    value = run_contact_case(
                        config,
                        case,
                        backend=backend,
                        cleanup_callback=lambda c: _write(
                            root, f"cleanup/contact-{index:02d}.json", c
                        ),
                        evidence_callback=failure_artifact,
                    )
                    summary = value["summary"]
                    if case.get("dt"):
                        original = next(
                            c
                            for c in report["cases"]
                            if c["name"] == case["name"].replace("_half_dt", "_positive")
                        )
                        expected = original["summary"]["signed_load_displacement_m"]
                        actual = summary["signed_load_displacement_m"]
                        same = all(
                            summary["gates"].get(k) == v
                            for k, v in original["summary"]["gates"].items()
                        )
                        # At rest compare absolute slip, since a relative zero denominator is meaningless.
                        agree = abs(actual - expected) <= max(0.0001, 0.1 * abs(expected))
                        summary["gates"]["half_dt_classification_and_displacement"] = same and agree
                        summary["passed"] = all(summary["gates"].values())
                    ref = save_bundle(root, f"contact-{index:02d}", value)
                    record = {"name": case["name"], "outcome": ref, "summary": summary}
                    report["cases"].append(record)
                    receipt["outcome"] = ref
                    _write(root, "contact-report.json", report)
                    if not summary["passed"]:
                        report.update(status="contact_limitation", stage_passed=False)
                        break
            else:
                report.update(status="contact_passed", stage_passed=True)
    except BaseException as exc:
        report.update(status="stopped", stage_passed=False, error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            raise
    finally:
        _finish(root, "contact-report.json", report, budget)
    return report


def run_diagonal_campaign(output_dir, *, contact_report=None, backend="gpu"):
    root = Path(output_dir).resolve()
    if (root / "diagonal-report.json").exists():
        return terminal_report(root, "diagonal-report.json")
    contact_path = root / "contact-report.json"
    if contact_report is not None and Path(contact_report).resolve() != contact_path:
        raise ValueError("motion must share the contact campaign directory and ledger")
    contact = terminal_report(root, "contact-report.json")
    if (
        contact is None
        or contact.get("status") != "contact_passed"
        or not contact.get("stage_passed")
    ):
        return {
            "status": "contact_gate_stop",
            "stage_passed": False,
            "contact_status": None if contact is None else contact["status"],
        }
    if (root / "diagonal-stage.claim").exists():
        return {"status": "uncertain_incomplete", "stage_passed": False}
    budget = DiagonalBudget(root)
    _exclusive_claim(
        root / "diagonal-stage.claim", {"contact_report_sha256": file_sha256(contact_path)}
    )
    report = {
        "schema": "cloth_fling.reference_diagonal.v3",
        "status": "started",
        "stage_passed": False,
        "contact_report_sha256": file_sha256(contact_path),
        "attempts": [],
        "frozen_candidate": None,
        "learning": "not_evaluated",
        "varied_grasps": "not_evaluated",
        "material_fidelity": "not_evaluated",
    }
    _write(root, "diagonal-report.json", report)
    env = None
    try:
        # Sources and configuration must remain the exact frozen qualification implementation.
        for ref, digest in contact["source_files"].items():
            current = Path(__file__).parent / Path(ref).relative_to("source")
            if file_sha256(current) != digest:
                raise RuntimeError("source changed after contact qualification")
        if file_sha256(root / "configuration.json") != contact["configuration_sha256"]:
            raise RuntimeError("frozen configuration changed after contact qualification")
        config_doc = json.loads((root / "configuration.json").read_text())
        state = read_hashed_bundle(root, contact["fixture"]["path"])
        if file_sha256(root / contact["fixture"]["path"]) != contact["fixture"]["sha256"]:
            raise RuntimeError("fixture metadata hash mismatch")
        obs = read_hashed_bundle(root, contact["fixture_observation"]["path"])
        if (
            file_sha256(root / contact["fixture_observation"]["path"])
            != contact["fixture_observation"]["sha256"]
        ):
            raise RuntimeError("fixture observation hash mismatch")
        action = GraspAction(**config_doc["favorable_action"])
        with reference_lease(root):
            try:
                from cloth_fling.reference_diagonal_motion import DiagonalTrajectory
                from cloth_fling.sim.appearance import AppearanceConfig
                from cloth_fling.sim.reference import ReferenceBackend

                with budget.reserve("preparation", 60):
                    env = ReferenceBackend(
                        diagonal_scene(), backend=backend, appearance=AppearanceConfig()
                    )
                    if env.diagnostic_field is not None:
                        raise RuntimeError("diagnostic force leaked into motion")
                    restore_exact(env, state, obs["rgb"], obs["observation_id"])

                def attempt(candidate, category):
                    number = len(report["attempts"])
                    with budget.reserve(
                        category, 180, attempt=number, candidate=candidate["index"]
                    ) as receipt:
                        restore_exact(env, state, obs["rgb"], obs["observation_id"])
                        env.take_contact_accounting(reset=True)
                        value = execute_diagonal_action(
                            env,
                            action,
                            DiagonalTrajectory(**candidate["trajectory"]),
                            expected_ids=config_doc["expected_ids"],
                            cleanup_callback=lambda c: _write(
                                root, f"cleanup/action-{number:02d}.json", c
                            ),
                        )
                        gates = acceptance(value)
                        ref = save_bundle(root, f"action-{number:02d}", value)
                        record = {
                            "attempt": number,
                            "category": category,
                            "candidate": candidate["index"],
                            "outcome": ref,
                            "gates": gates,
                            "passed": all(gates.values()),
                            "reason": value.reason,
                            "coverage_after": value.coverage_after,
                            "simulated_seconds": value.simulated_seconds,
                            "diagnostics": {
                                k: v for k, v in value.diagnostics.items() if k != "tauten"
                            },
                        }
                        report["attempts"].append(record)
                        receipt["outcome"] = ref
                        _write(root, "diagonal-report.json", report)
                        return record, campaign_stopping(value)

                for candidate in config_doc["motion_grid"]:
                    record, fatal = attempt(candidate, "development")
                    if fatal:
                        report["status"] = "campaign_stopping_failure"
                        break
                    if record["passed"]:
                        report["frozen_candidate"] = candidate
                        _write(root, "frozen-trajectory.json", candidate)
                        _write(root, "diagonal-report.json", report)
                        passed = True
                        for _ in range(3):
                            check, fatal = attempt(candidate, "confirmation")
                            passed &= check["passed"]
                            if fatal:
                                report["status"] = "campaign_stopping_failure"
                                break
                        else:
                            report.update(
                                status="complete" if passed else "confirmation_incomplete",
                                stage_passed=passed,
                            )
                        break
                else:
                    report["status"] = "placement_limitation"
            finally:
                if env is not None:
                    _cleanup(env, root, "cleanup/final.json")
    except BaseException as exc:
        report.update(status="stopped", stage_passed=False, error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            raise
    finally:
        _finish(root, "diagonal-report.json", report, budget)
    return report
