"""Post-grasp lifecycle and frozen-control parity; fake dynamics do not qualify behavior."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_reference_diagonal import DemonstrationFake

from cloth_fling.contracts import GraspAction
from cloth_fling.frozen_laydown import (
    CONFIG_NAME,
    VERSION,
    execute_frozen_laydown,
    frozen_laydown_spec,
)
from cloth_fling.reference_diagonal_motion import (
    DiagonalTrajectory,
    diagonal_schedule,
    execute_diagonal_action,
)
from cloth_fling.reference_surface import diagonal_scene

ACTION = GraspAction(((0, 0), (1, 1)))


class HandoffFake(DemonstrationFake):
    def __init__(self):
        super().__init__()
        self.config = diagonal_scene()

    def reset(self, *args):
        pytest.fail("handoff reset the scene")

    def restore(self, *args):
        pytest.fail("handoff restored an archived fixture")


def held_env():
    env = HandoffFake()
    env.attach(ACTION.pixels)
    env.simulated_time = 42.0
    env.attach = lambda *args: pytest.fail("handoff reattached or reselected the grasp")
    return env


def test_existing_grasp_handoff_matches_frozen_motion_without_reset_or_reattachment():
    baseline = HandoffFake()
    baseline.simulated_time = 42.0
    reference = execute_diagonal_action(
        baseline,
        ACTION,
        DiagonalTrajectory(),
        expected_ids=(0, 20),
        cleanup_callback=lambda value: None,
        execution_mode="demonstration",
    )
    env = held_env()
    certificates = []
    result = execute_frozen_laydown(env, ACTION, cleanup_callback=certificates.append)
    assert result.execution_valid and result.reason == "completed"
    assert certificates[0]["passed"] and not env.attached_ids
    assert result.before.observation_id == "42.0"
    assert env.simulated_time == pytest.approx(42 + result.simulated_seconds)
    assert result.evidence["retained_existing_grasp"]
    assert result.evidence["native_states"][0]["state"]["free"].sum() == len(env.p) - 2
    assert result.diagnostics["frozen_laydown_version"] == VERSION
    handoff = result.evidence["frozen_laydown"]
    assert handoff["incoming_native_pin_ids"] == [0, 20]
    assert handoff["incoming_simulated_time"] == 42
    assert result.simulated_seconds == pytest.approx(reference.simulated_seconds)
    np.testing.assert_allclose(env.p, baseline.p, atol=1e-11)
    commands = result.evidence["commands"]
    expected = reference.evidence["commands"]
    assert len(commands) == len(expected)
    for actual, old in zip(commands, expected, strict=True):
        assert actual["phase"] == ("grasp_handoff" if old["phase"] == "attach" else old["phase"])
        if old["commanded_grasp_positions_m"] is not None:
            np.testing.assert_allclose(
                actual["commanded_grasp_positions_m"],
                old["commanded_grasp_positions_m"],
                atol=1e-11,
            )


@pytest.mark.parametrize("ids", [(), (0,), (0, 0), (0, 20, 24), (20, 0)])
def test_invalid_handoff_leaves_the_callers_grasp_and_state_untouched(ids):
    env = held_env()
    env.attached_ids = ids
    before = env.p.copy()
    calls = []
    with pytest.raises(ValueError, match="handoff"):
        execute_frozen_laydown(env, ACTION, cleanup_callback=calls.append)
    assert env.attached_ids == ids and env.simulated_time == 42
    np.testing.assert_array_equal(env.p, before)
    assert calls == []


def test_hidden_native_pin_or_physics_mismatch_rejects_without_releasing_the_grasp():
    env = held_env()
    free = env._free()
    free[5] = False
    env._free = lambda: free
    with pytest.raises(ValueError, match="native pins"):
        execute_frozen_laydown(env, ACTION, cleanup_callback=lambda c: None)
    assert env.attached_ids == (0, 20) and env.simulated_time == 42
    env = held_env()
    env.config = replace(env.config, sim_dt=0.0005)
    with pytest.raises(ValueError, match="physical configuration"):
        execute_frozen_laydown(env, ACTION, cleanup_callback=lambda c: None)
    assert env.attached_ids == (0, 20) and env.simulated_time == 42
    env.config = diagonal_scene()
    env.diagnostic_field = object()
    with pytest.raises(ValueError, match="external loads"):
        execute_frozen_laydown(env, ACTION, cleanup_callback=lambda c: None)
    assert env.attached_ids == (0, 20)


def test_execution_failure_after_adoption_releases_existing_pins_and_certifies_cleanup():
    env = held_env()
    step = env.step

    def fail_step():
        if env.simulated_time > 42.04:
            raise RuntimeError("injected simulator failure")
        step()

    env.step = fail_step
    certificates = []
    result = execute_frozen_laydown(env, ACTION, cleanup_callback=certificates.append)
    assert not result.execution_valid and result.reason == "injected simulator failure"
    assert certificates[0]["passed"] and env._free().all()
    assert result.evidence["frozen_laydown"]["incoming_native_pin_ids"] == [0, 20]


def test_bad_api_arguments_or_changed_implementation_do_not_take_grasp_ownership(monkeypatch):
    env = held_env()
    for action, callback in ((None, lambda c: None), (ACTION, None)):
        with pytest.raises(TypeError, match="GraspAction"):
            execute_frozen_laydown(env, action, cleanup_callback=callback)
        assert env.attached_ids == (0, 20)
    import cloth_fling.frozen_laydown as frozen

    spec = frozen_laydown_spec()
    spec["implementation_sha256"]["reference_motion.py"] = "0" * 64
    monkeypatch.setattr(frozen, "frozen_laydown_spec", lambda: spec)
    with pytest.raises(ValueError, match="implementation changed"):
        execute_frozen_laydown(env, ACTION, cleanup_callback=lambda c: None)
    assert env.attached_ids == (0, 20) and env.simulated_time == 42


def test_frozen_artifacts_match_the_selected_trajectory_and_packaged_copy():
    spec = frozen_laydown_spec()
    root = Path(__file__).parents[1]
    assert (root / "configs" / CONFIG_NAME).read_bytes() == (
        root / "src/cloth_fling/profiles" / CONFIG_NAME
    ).read_bytes()
    assert spec["version"] == VERSION
    assert spec["trajectory"]["endpoint_x_m"] == 0.78
    assert spec["trajectory"]["diagonal_seconds"] == 6
    assert spec["schedule"] == diagonal_schedule(DiagonalTrajectory())[1]
    assert spec["scene"] == json.loads(json.dumps(diagonal_scene().to_dict()))
    assert not spec["reference_evidence"]["qualification_complete"]
    assert spec["reference_evidence"]["coverage_after"] == pytest.approx(0.9797666666666668)
    spec["trajectory"]["diagonal_seconds"] = 4
    assert frozen_laydown_spec()["trajectory"]["diagonal_seconds"] == 6


def test_freeze_loader_is_offline_and_independent_of_the_current_working_directory(tmp_path):
    code = "import sys; from cloth_fling.frozen_laydown import frozen_laydown_spec; assert frozen_laydown_spec()['version']=='reference_post_grasp_laydown_v1'; assert 'genesis' not in sys.modules and 'torch' not in sys.modules"
    import os

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
