"""Off-center physical target commands, not just the standalone tug polynomial."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_reference_diagonal import DemonstrationFake

from cloth_fling.frozen_laydown import frozen_laydown_spec
from cloth_fling.reference_diagonal_motion import DiagonalMotion
from cloth_fling.reference_pilot import audit_tug_commands
from cloth_fling.reference_trajectory import Quintic


@pytest.mark.parametrize("midpoint_y", [0.0, -0.0472378134727478, -0.10643864050507545, 0.137])
def test_all_twenty_tug_increments_preserve_yz_and_bound_native_substep_commands(midpoint_y):
    env = DemonstrationFake()
    run = DiagonalMotion(env, execution_mode="demonstration")
    run.ids = env.attach(None)
    initial = np.array([[0.78, midpoint_y - 0.1, 0.31], [0.78, midpoint_y + 0.1, 0.31]])
    run.target = initial.copy()
    env.p[list(run.ids)] = initial
    # Hold and cloth feedback are isolated; execute the actual curve and native target path.
    run.hold = lambda *args: None
    run.sample = lambda *args: None
    run.body.freeze_tug = lambda *args, **kwargs: None
    targets = [initial.copy()]
    set_targets = env.set_attachment_targets

    def record(target):
        targets.append(target.copy())
        set_targets(target)

    env.set_attachment_targets = record
    run.perform_tug()
    targets = np.asarray(targets)
    assert len(targets) == 2001
    assert run.tug_distance == pytest.approx(0.02)
    assert run.tug_stop_reason == "distance_cap"
    np.testing.assert_allclose(
        targets[:, :, 1:], np.broadcast_to(initial[:, 1:], targets[:, :, 1:].shape), atol=1e-12
    )
    np.testing.assert_allclose(targets[-1], initial + [0.02, 0, 0], atol=1e-12)
    velocity = (
        np.diff(np.concatenate([targets[:1], targets, targets[-1:]]), axis=0) / env.config.sim_dt
    )
    acceleration = np.diff(velocity, axis=0) / env.config.sim_dt
    assert np.linalg.norm(velocity, axis=2).max() <= 0.01875 + 1e-8
    assert np.linalg.norm(acceleration, axis=2).max() <= 0.578
    # The very first native substep is bounded, not an instantaneous midpoint-Y shift.
    assert np.linalg.norm(targets[1] - targets[0], axis=1).max() < 1e-7


@pytest.mark.parametrize("midpoint_y", [-0.0472378134727478, -0.10643864050507545, 0.137])
def test_saved_command_assessment_rejects_the_original_lateral_jump(midpoint_y):
    initial = np.array([[0.78, midpoint_y - 0.1, 0.31], [0.78, midpoint_y + 0.1, 0.31]])
    segment = Quintic.between("tug", 0.1, [0, 0, 0], [0.001, 0, 0], [0, 0, 0], [0, 0, 0])
    commands = [
        {
            "phase": "pre_tug_stabilize",
            "simulated_seconds": 0.0,
            "commanded_grasp_positions_m": initial.copy(),
        }
    ]
    for tick in range(1, 7):
        commands.append(
            {
                "phase": "tug" if tick <= 5 else "endpoint_hold",
                "simulated_seconds": tick * 0.02,
                "commanded_grasp_positions_m": initial + segment.at(min(tick * 0.02, 0.1)),
            }
        )
    assert audit_tug_commands(commands)["passed"]
    for command in commands[1:]:
        command["commanded_grasp_positions_m"][:, 1] += midpoint_y
    audit = audit_tug_commands(commands)
    assert not audit["passed"]
    assert audit["max_lateral_target_error_m"] == pytest.approx(abs(midpoint_y))
    assert not audit_tug_commands([])["passed"]


def test_v2_preserves_v1_archive_and_changes_only_tug_frame_and_implementation_identity():
    root = Path(__file__).parents[1]
    old_path = root / "configs/reference-post-grasp-laydown-v1-frozen.json"
    assert (
        hashlib.sha256(old_path.read_bytes()).hexdigest()
        == "8324d77b515b32f5a85689462e568e6a24c629fba3eb9ae8b9586bb3917c317f"
    )
    assert old_path.read_bytes() == (root / "src/cloth_fling/profiles" / old_path.name).read_bytes()
    old, new = json.loads(old_path.read_text()), frozen_laydown_spec()
    for key in ("scene", "trajectory", "schedule", "reference_evidence", "handoff"):
        assert new[key] == old[key]
    assert new["controller"].pop("tug_coordinate_frame") == "midpoint_relative_Y"
    assert new["controller"] == old["controller"]
    changed = [
        name
        for name, value in new["implementation_sha256"].items()
        if value != old["implementation_sha256"][name]
    ]
    assert changed == ["reference_diagonal_motion.py"]
    assert new["version"] == "reference_post_grasp_laydown_v2"
