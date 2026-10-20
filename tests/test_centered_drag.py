"""Geometry-based development centering and cumulative wall accounting."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_fling.contracts import ReferenceSceneConfig
from cloth_fling.reference_diagonal_motion import centered_placement_segments
from cloth_fling.reference_motion import ReferenceMotionFailure
from cloth_fling.reference_pilot import run_pilot


@pytest.mark.parametrize(
    "drop,body_y,mid_y",
    [
        (0.493781, -0.034248263, -0.04723781),
        (0.33662993, -0.08895071, -0.13156708),
        (0.4, 0.1, 0.08),
    ],
)
def test_measured_geometry_changes_laydown_without_reintroducing_drag_join_jump(
    drop, body_y, mid_y
):
    target = np.array([[0.1, mid_y - 0.1, 0.95], [0.1, mid_y + 0.1, 0.95]])
    positions = np.array([[0.1, body_y - 0.2, 0.95 - drop], [0.1, body_y + 0.2, 0.95]])
    env = SimpleNamespace(config=ReferenceSceneConfig(), get_positions=lambda: positions)
    (diagonal, drag), metadata = centered_placement_segments(env, target)
    assert diagonal.at(6)[0] < 0.73
    assert drag.at(2)[0] - drag.at(0)[0] == pytest.approx(0.12)
    offset = target - target.mean(0)
    offset[:, 1] += mid_y
    diagonal_end = diagonal.at(6) + offset
    next_offset = diagonal_end - diagonal_end.mean(0)
    next_offset[:, 1] += diagonal_end[:, 1].mean()
    np.testing.assert_allclose(drag.at(0) + next_offset, diagonal_end, atol=1e-12)
    np.testing.assert_allclose((drag.at(2) + next_offset)[:, 1:], diagonal_end[:, 1:], atol=1e-12)
    assert metadata["planned_laydown_Y_displacement_m"] == pytest.approx(-body_y)
    assert metadata["measured_hanging_drop_m"] == pytest.approx(drop)
    for s in (diagonal, drag):
        assert s.maximum_norm(1) <= 0.8 and s.maximum_norm(2) <= 2
        for derivative in (1, 2):
            np.testing.assert_allclose(s.at(0, derivative), 0, atol=1e-12)
            np.testing.assert_allclose(s.at(s.seconds, derivative), 0, atol=1e-12)


def test_outside_centering_target_is_rejected_before_issuing_any_targets():
    env = SimpleNamespace(
        config=ReferenceSceneConfig(),
        get_positions=lambda: np.array([[0.1, 0, -0.09], [0.1, 0.1, 0.95]]),
    )
    target = np.array([[0.1, -0.1, 0.95], [0.1, 0.1, 0.95]])
    with pytest.raises(ReferenceMotionFailure, match="centering_target_outside_table"):
        centered_placement_segments(env, target)


@pytest.mark.parametrize(
    "status,wall", [("started_never_resume", 105), ("failed_or_uncertain", 105), ("complete", 3600)]
)
def test_uncertain_or_exhausted_predecessor_cannot_open_a_scene(tmp_path, status, wall):
    predecessor = tmp_path / "prior.json"
    predecessor.write_text(json.dumps({"status": status, "wall_seconds": wall}))
    with pytest.raises(ValueError, match="terminal, accounted"):
        run_pilot(
            tmp_path / "run",
            tmp_path / "missing-fixture",
            budget_predecessor=predecessor,
            backend_factory=lambda *a: pytest.fail("opened scene"),
        )
