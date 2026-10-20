"""Offline safeguards; fake dynamics cannot qualify towel behavior."""

import json

import numpy as np
import pytest
from test_reference import FakeReference

from cloth_fling.contracts import GraspAction
from cloth_fling.reference_leading_motion import (
    LeadingMotion,
    acceptance,
    campaign_stopping,
    execute_leading_action,
)
from cloth_fling.reference_trajectory import (
    LeadingTrajectory,
    candidate_grid,
    leading_fraction,
    placement_schedule,
)
from cloth_fling.reference_tune import TuneBudget, restore_exact, run_reference_tuning


class NativeFake(FakeReference):
    def snapshot(self):
        return {**super().snapshot(), "velocities": self.v.copy()}


def test_grid_and_all_analytic_extrema_and_continuous_joins():
    grid = candidate_grid()
    assert [(c.transfer_x_m, c.descent_seconds) for c in grid] == [
        (x, t) for x in (0.45, 0.55, 0.35, 0.65) for t in (2, 3, 4, 1.5, 1)
    ]
    for config in grid:
        if config.transfer_x_m == 0.65 and config.descent_seconds in (3, 4):
            with pytest.raises(ValueError, match="horizontal reversal"):
                placement_schedule(config)
            continue
        segments, schedule = placement_schedule(config)
        assert 0 <= schedule["tauten_allowance_seconds"] <= 20
        assert schedule["maximum_action_seconds"] <= 35 + 1e-8
        for segment in segments:
            assert segment.minimum_horizontal_speed() >= -1e-10
            assert segment.maximum_norm(1) <= 0.8 + 1e-9
            assert segment.maximum_norm(2) <= 2 + 1e-9
            # Independent dense sample verifies that analytical roots include maxima.
            t = np.linspace(0, segment.seconds, 2001)
            for derivative in (1, 2):
                assert (
                    np.linalg.norm(segment.at(t, derivative), axis=0).max()
                    <= segment.maximum_norm(derivative) + 1e-8
                )
        for left, right in zip(segments, segments[1:]):
            for derivative in (0, 1, 2):
                np.testing.assert_allclose(
                    left.at(left.seconds, derivative), right.at(0, derivative), atol=1e-12
                )
            assert left.at(left.seconds, 1)[0] > 0
        np.testing.assert_allclose(
            segments[-1].at(segments[-1].seconds), [0.8, 0, 0.31], atol=1e-12
        )
        np.testing.assert_allclose(segments[-1].at(segments[-1].seconds, 1), 0, atol=1e-12)


def test_leading_excludes_pins_and_allows_only_two_cm():
    positions = np.array([[0.9, 0, 0], [0.81, 0, 0], [0.821, 0, 0], [0.7, 0, 0]])
    free = np.array([False, True, True, True])
    assert leading_fraction(positions, free, 0.8) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        leading_fraction(positions, np.zeros(4, bool), 0.8)


def test_reserved_tauten_cannot_consume_placement_budget():
    env = NativeFake()
    run = LeadingMotion(env)
    run.ids = env.attach(None)
    run.target = env.get_positions()[list(run.ids)]
    run.target[:, 2] = 0.95
    env.p[:, 2] = 0.95
    allowance = placement_schedule(LeadingTrajectory())[1]["tauten_allowance_seconds"]
    begin = env.simulated_time
    run.tauten_cloth(allowance=0.02)
    assert env.simulated_time - begin <= 0.02 + 1e-8
    assert allowance < 20


def test_native_early_edge_failure_and_cleanup_are_durable():
    class Crossing(NativeFake):
        def step(self):
            super().step()
            # Inject ordinary far-edge placement failure while keeping mesh finite.
            if self.simulated_time > 7 and self.p[list(self.attached_ids), 2].mean() < 0.32:
                self.p[24] = [0.951, 0.2, 0.31]

    env = Crossing()
    cleanup = []
    result = execute_leading_action(
        env,
        GraspAction(((0, 0), (1, 1))),
        LeadingTrajectory(),
        expected_ids=(0, 20),
        cleanup_callback=cleanup.append,
    )
    assert result.reason == "placement_far_edge_crossing", result.reason
    assert cleanup[0]["passed"]
    assert not campaign_stopping(result)
    assert not all(acceptance(result).values())
    states = [e["event"] for e in result.evidence["native_states"]]
    assert states[0] == "time_zero" and "failure" in states and states[-1] == "termination"
    assert result.evidence["phase_traces"][0]["simulated_seconds"] == 0
    assert not env.attached_ids


def test_favorable_vertex_change_stops_campaign():
    cleanup = []
    result = execute_leading_action(
        NativeFake(),
        GraspAction(((0, 0), (1, 1))),
        LeadingTrajectory(),
        expected_ids=(1, 20),
        cleanup_callback=cleanup.append,
    )
    assert campaign_stopping(result)
    assert cleanup[0]["passed"]
    assert result.reason == "favorable_vertex_identity_mismatch"


def test_exact_restore_rejects_image_or_hash_drift():
    class Env:
        def restore(self, state):
            return type("Obs", (), dict(observation_id="exact", rgb=np.zeros((2, 2, 3))))()

        def snapshot(self):
            return {"state_hash": "same"}

    env = Env()
    restore_exact(env, {"state_hash": "same"}, np.zeros((2, 2, 3)), "exact")
    for state, rgb, oid in [
        ({"state_hash": "drift"}, np.zeros((2, 2, 3)), "exact"),
        ({"state_hash": "same"}, np.ones((2, 2, 3)), "exact"),
        ({"state_hash": "same"}, np.zeros((2, 2, 3)), "other"),
    ]:
        with pytest.raises(RuntimeError, match="regression"):
            restore_exact(env, state, rgb, oid)


def test_durable_limits_and_failed_reservations_cannot_resume(tmp_path):
    budget = TuneBudget(tmp_path)
    for _ in range(20):
        with budget.reserve("development", 1):
            document = json.loads((tmp_path / "tune-budget.json").read_text())
            assert document["operations"][-1]["status"] == "reserved"
    with pytest.raises(RuntimeError, match="attempt budget"):
        with budget.reserve("development", 1):
            pass
    for _ in range(3):
        with budget.reserve("confirmation", 1):
            pass
    with pytest.raises(RuntimeError, match="attempt budget"):
        with budget.reserve("confirmation", 1):
            pass
    with pytest.raises(ValueError):
        with budget.reserve("preparation", 10):
            raise ValueError("uncertain")
    document = json.loads((tmp_path / "tune-budget.json").read_text())
    assert document["operations"][-1]["status"] == "failed_or_uncertain"
    assert document["operations"][-1]["charged_seconds"] == 10
    with pytest.raises(RuntimeError, match="cannot resume"):
        TuneBudget(tmp_path)
    (tmp_path / "tune-report.json").write_text(json.dumps({"status": "started"}))
    assert (
        run_reference_tuning(tmp_path, fixture_dir=tmp_path / "does-not-exist")["status"]
        == "uncertain_incomplete"
    )
    (tmp_path / "tune-report.json").write_text(
        json.dumps({"status": "placement_limitation", "stage_passed": False})
    )
    assert (
        run_reference_tuning(tmp_path, fixture_dir=tmp_path / "does-not-exist")["status"]
        == "placement_limitation"
    )


def test_wall_budget_rejects_before_execution(tmp_path):
    budget = TuneBudget(tmp_path)
    budget.document["operations"].append(
        {
            "category": "preparation",
            "allowance_seconds": 7190,
            "status": "failed_or_uncertain",
            "charged_seconds": 7190,
        }
    )
    with pytest.raises(RuntimeError, match="wall budget"):
        with budget.reserve("development", 180):
            pytest.fail("must reject offline")


def archived_campaign(root):
    from dataclasses import asdict

    from cloth_fling.contracts import ReferenceSceneConfig
    from cloth_fling.reference_run import save_bundle

    root.mkdir()
    before = {"rgb": np.zeros((2, 2, 3), dtype=np.uint8), "observation_id": "fixture"}
    fixture = {
        "schema_version": 2,
        "state_hash": "same",
        "config_json": json.dumps(ReferenceSceneConfig().to_dict()),
    }
    ref = save_bundle(root, "fixture", fixture)
    previous = save_bundle(
        root,
        "single-action",
        {
            "action": asdict(GraspAction(((0, 0), (1, 1)))),
            "before": before,
            "evidence": {"commands": [{"attachment_ids": [0, 20]}]},
        },
    )
    (root / "reference-report.json").write_text(
        json.dumps(
            {
                "status": "placement_limitation",
                "restoration_passed": True,
                "contact": {"passed": True},
                "initial_state": ref,
                "single_action": previous,
            }
        )
    )


@pytest.mark.parametrize("confirmation_failure", [False, True])
def test_first_pass_frozen_and_confirmations_never_tune(
    tmp_path, monkeypatch, confirmation_failure
):
    from contextlib import nullcontext

    import cloth_fling.reference_tune as tune
    from cloth_fling.sim.reference import ReferenceBackend

    archive = tmp_path / "archive"
    archived_campaign(archive)
    original_files = {p.name: p.read_bytes() for p in archive.iterdir()}
    monkeypatch.setattr(tune, "reference_lease", lambda root: nullcontext())

    class Env:
        def __init__(self, *args, **kwargs):
            from cloth_fling.contracts import ReferenceSceneConfig

            self.config = ReferenceSceneConfig()
            self.rest_positions = np.zeros((3, 3))
            self.cloth = self
            self.attached_ids = ()

        def set_particles_pos(self, p):
            pass

        def set_particles_vel(self, v):
            pass

        def take_contact_accounting(self, **kwargs):
            return {"corrected_particles": 1}

        def step(self, n):
            pass

        def get_velocities(self):
            return np.zeros((3, 3))

        def snapshot(self):
            return {"state_hash": "same"}

        def release(self):
            pass

        def _free(self):
            return np.ones(3, bool)

        def close(self):
            pass

    monkeypatch.setattr(ReferenceBackend, "__new__", staticmethod(lambda cls, *a, **k: Env()))
    monkeypatch.setattr(tune, "restore_exact", lambda *a: None)
    calls = []

    def execute(*args, cleanup_callback, **kwargs):
        calls.append(kwargs["expected_ids"])
        cleanup_callback({"passed": True})

        class Outcome:
            reason = "completed"
            diagnostics = {}
            coverage_after = 1.0
            simulated_seconds = 20

        return Outcome()

    monkeypatch.setattr(tune, "execute_leading_action", execute)
    monkeypatch.setattr(
        tune, "acceptance", lambda value: {"all": not (confirmation_failure and len(calls) == 2)}
    )
    monkeypatch.setattr(tune, "campaign_stopping", lambda value: False)
    original_save = tune.save_bundle
    monkeypatch.setattr(
        tune,
        "save_bundle",
        lambda root, name, value: (
            original_save(root, name, {"reason": "fake"})
            if name.startswith("attempt")
            else original_save(root, name, value)
        ),
    )
    report = run_reference_tuning(tmp_path / "run", fixture_dir=archive)
    assert report["status"] == (
        "confirmation_incomplete" if confirmation_failure else "complete"
    ), report
    assert len(calls) == 4
    assert report["frozen_candidate"]["index"] == 0
    assert all(a["candidate"] == 0 for a in report["attempts"])
    assert (tmp_path / "run" / "frozen-trajectory.json").exists()
    assert {p.name: p.read_bytes() for p in archive.iterdir()} == original_files
    # The exact output returns its terminal report offline.
    assert run_reference_tuning(tmp_path / "run", fixture_dir=archive)["status"] == report["status"]
    with pytest.raises(RuntimeError, match="already has"):
        run_reference_tuning(tmp_path / "another-output", fixture_dir=archive)


def test_tuning_replay_preserves_failure_and_rejects_corruption(tmp_path):
    from test_replay import snapshot, traces

    from cloth_fling.reference_run import save_bundle
    from cloth_fling.replay import load_replay, sample_traces

    save_bundle(tmp_path, "fixture", snapshot())
    ref = save_bundle(
        tmp_path,
        "attempt-00",
        {
            "execution_valid": False,
            "reason": "placement_far_edge_crossing",
            "simulated_seconds": 1.0,
            "evidence": {"phase_traces": traces()},
        },
    )
    (tmp_path / "tune-report.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {"attempt": 0, "category": "development", "candidate": 0, "outcome": ref}
                ]
            }
        )
    )
    actions, origin = load_replay(tmp_path)
    assert origin["invalid_attempts"] == 1
    assert actions[0].reason == "placement_far_edge_crossing"
    assert list(sample_traces(actions[0], 10))[0]["simulated_seconds"] == 0
    (tmp_path / ref["path"]).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_replay(tmp_path)


def test_historical_reference_motion_cannot_extend(tmp_path):
    from cloth_fling.reference_run import ReferenceBudget, run_reference_qualification

    previous = tmp_path / "old"
    previous.mkdir()
    budget = ReferenceBudget(previous)
    with budget.reserve("diagnostic", 1):
        pass
    (previous / "reference-report.json").write_text('{"status":"placement_limitation"}')
    # This rejection happens before importing/constructing the native scene.
    with pytest.raises(ValueError, match="terminal motion evidence"):
        run_reference_qualification(tmp_path / "next", previous=previous)
    assert not (tmp_path / "next" / "gpu-lease.json").exists()
    with pytest.raises(FileExistsError):
        run_reference_qualification(previous)


def test_versioned_configuration_matches_executed_grid():
    from dataclasses import asdict
    from pathlib import Path

    config = json.loads(
        (Path(__file__).parents[1] / "configs/reference-leading-v2.json").read_text()
    )
    assert config["candidates"] == [asdict(c) for c in candidate_grid()]
    assert config["limits"] == {
        "development": 20,
        "confirmation": 3,
        "simulator_wall_seconds": 7200,
        "action_simulated_seconds": 35,
        "action_wall_seconds": 180,
    }
    frozen = json.loads(
        (Path(__file__).parents[1] / "configs/reference-leading-v2-frozen.json").read_text()
    )
    _, schedule = placement_schedule(LeadingTrajectory(**frozen["trajectory"]))
    assert schedule == frozen["schedule"]


def test_terminal_tuning_cli_is_offline(tmp_path):
    import subprocess
    import sys

    (tmp_path / "tune-report.json").write_text(
        '{"status":"placement_limitation","stage_passed":false}'
    )
    script = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'genesis','torch'}:
            raise AssertionError('offline execution imported '+fullname)
sys.meta_path.insert(0, Guard())
from cloth_fling.cli import main
assert main(['reference','tune','--output',sys.argv[1],'--fixture','missing']) == 1
assert 'genesis' not in sys.modules and 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
