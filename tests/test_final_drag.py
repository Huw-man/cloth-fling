"""A continuous one-fifth-length drag, including endpoint joins and execution evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_reference_diagonal import DemonstrationFake

from cloth_fling.frozen_laydown import frozen_laydown_spec
from cloth_fling.reference_diagonal_motion import (
    DiagonalMotion,
    DiagonalTrajectory,
    diagonal_schedule,
)
from cloth_fling.reference_pilot import audit_tug_commands


def trajectory():
    return DiagonalTrajectory(
        version="reference_diagonal_drag_v4", endpoint_x_m=0.73, max_tug_m=0.12
    )


def test_schedule_reserves_full_drag_without_extending_action_or_tautening_budget():
    segments, schedule = diagonal_schedule(trajectory())
    diagonal, drag = segments
    assert schedule["maximum_action_seconds"] == pytest.approx(35)
    assert schedule["tauten_allowance_seconds"] == pytest.approx(14.6)
    assert drag.seconds == 2
    np.testing.assert_allclose(drag.at(0), diagonal.at(diagonal.seconds), atol=1e-12)
    np.testing.assert_allclose(drag.at(2), [0.85, 0, 0.31], atol=1e-12)
    for derivative in (1, 2):
        np.testing.assert_allclose(drag.at(0, derivative), 0, atol=1e-12)
        np.testing.assert_allclose(drag.at(2, derivative), 0, atol=1e-12)
    assert drag.maximum_norm(1) == pytest.approx(0.1125)
    assert drag.maximum_norm(2) == pytest.approx(0.1732050807568877)
    assert 0.95 - drag.at(2)[0] == pytest.approx(0.10)


@pytest.mark.parametrize("midpoint_y", [-0.1064386405, 0.0, 0.137])
def test_drag_completes_full_distance_despite_slack_or_supported_body_sliding(midpoint_y):
    env = DemonstrationFake()
    run = DiagonalMotion(env, execution_mode="demonstration")
    run.ids = env.attach(None)
    initial = np.array([[0.73, midpoint_y - 0.1, 0.31], [0.73, midpoint_y + 0.1, 0.31]])
    run.target = initial.copy()
    env.p[list(run.ids)] = initial
    run.prepare_final_contact = lambda: None
    run.sample = lambda *a: None
    run.slack_stable = 100  # Would immediately stop the old adaptive tug.
    run.body.tug_ids = np.flatnonzero(env._free())
    run.body.tug_anchors = env.p[run.body.tug_ids, :2] - 0.05
    targets = [initial.copy()]
    set_targets = env.set_attachment_targets

    def record(target):
        targets.append(target.copy())
        set_targets(target)

    env.set_attachment_targets = record
    run.perform_drag(diagonal_schedule(trajectory())[0][1])
    targets = np.asarray(targets)
    assert len(targets) == 2001 and run.tug_stop_reason == "completed_full_drag"
    assert run.tug_distance == pytest.approx(0.12)
    np.testing.assert_allclose(targets[-1], initial + [0.12, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        targets[:, :, 1:], np.broadcast_to(initial[:, 1:], targets[:, :, 1:].shape), atol=1e-12
    )
    velocity = np.diff(np.concatenate([targets[:1], targets, targets[-1:]]), axis=0) / 0.001
    acceleration = np.diff(velocity, axis=0) / 0.001
    assert np.linalg.norm(velocity, axis=2).max() <= 0.1125 + 1e-9
    assert np.linalg.norm(acceleration, axis=2).max() <= 0.174


def test_drag_assessment_requires_full_distance_and_does_not_accept_a_short_tug():
    segment = diagonal_schedule(trajectory())[0][1]
    target = np.array([[0, -0.15, 0], [0, 0.05, 0]])
    commands = [
        {
            "phase": "pre_tug_stabilize",
            "simulated_seconds": 0.0,
            "commanded_grasp_positions_m": segment.at(0) + target,
        }
    ]
    commands.extend(
        {
            "phase": "tug",
            "simulated_seconds": i * 0.02,
            "commanded_grasp_positions_m": segment.at(i * 0.02) + target,
        }
        for i in range(1, 101)
    )
    commands.append(
        {
            "phase": "endpoint_hold",
            "simulated_seconds": 2.02,
            "commanded_grasp_positions_m": segment.at(2) + target,
        }
    )
    assert audit_tug_commands(commands, target_distance_m=0.12)["passed"]
    assert not audit_tug_commands(commands[:10], target_distance_m=0.12)["passed"]
    assert not audit_tug_commands([], target_distance_m=0.12)["passed"]


def test_v3_preserves_prior_frozen_bytes_and_physics():
    root = Path(__file__).parents[1]
    path = root / "configs/reference-post-grasp-laydown-v2-frozen.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "604c16538dcb6e7dcbb7fac4f41c32a7c2a7fc8a931f1f68e4d338f6f46d19c2"
    )
    old = json.loads(path.read_text())
    new = frozen_laydown_spec()
    assert new["scene"] == old["scene"] and new["handoff"] == old["handoff"]
    assert new["version"] == "reference_post_grasp_laydown_v3"
    assert new["controller"]["final_drag_fraction_cloth_length"] == 0.2
