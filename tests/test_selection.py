from dataclasses import replace

import numpy as np
import pytest
from test_primitives_fake import FakeClothBackend

from cloth_fling.contracts import Observation, SceneConfig
from cloth_fling.selection import (
    NoValidPairError,
    SelectionConfig,
    feasible_candidates,
    foreground_mask,
    pixels_to_world,
    select_action,
)


def observation():
    return FakeClothBackend().observe()


def test_perspective_optical_depth_pixel_centers():
    obs = observation()
    uv = np.array([[20, 25], [35, 40]], dtype=float)
    world = pixels_to_world(obs, uv)
    assert world.shape == (2, 3)
    expected = np.column_stack(
        ((uv[:, 0] + 0.5 - 32) / 100, -(uv[:, 1] + 0.5 - 32) / 100, np.zeros(2))
    )
    depth = obs.depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    expected[:, :2] *= depth[:, None]
    expected[:, 2] = 1 - depth
    np.testing.assert_allclose(world, expected)
    translated = dict(obs.calibration)
    translated["camera_to_world"] = np.asarray(
        translated["camera_to_world"], dtype=float
    ).copy()
    translated["camera_to_world"][0, 3] = 0.2
    obs.calibration = translated
    np.testing.assert_allclose(pixels_to_world(obs, uv), expected + [0.2, 0, 0])


def test_gray_table_and_black_background_are_not_foreground():
    rgb = np.full((16, 16, 3), (64, 71, 77), dtype=np.uint8)
    rgb[:2] = 0
    rgb[5:11, 5:11] = [200, 60, 40]
    obs = Observation(
        "gray-table",
        rgb,
        np.ones((16, 16)),
        {"world_bounds_m": [-0.2, 0.2, -0.2, 0.2], "camera_height_m": 1.0},
    )
    mask = foreground_mask(obs)
    assert mask.sum() == 36
    assert not mask[:5].any() and not mask[:, :5].any()
    cfg = SelectionConfig(min_grasp_separation_m=0.03, max_grasp_separation_m=0.3)
    for seed in range(20):
        result = select_action(obs, cfg, rng=np.random.default_rng(seed))
        assert all(mask[int(v), int(u)] for u, v in result.pixels)
    obs.rgb = rgb.astype(float) / 255
    np.testing.assert_array_equal(foreground_mask(obs), mask)


@pytest.mark.parametrize("policy", ["uniform_valid", "geometry_guided"])
def test_distinct_feasible_deterministic_visible_pairs(policy):
    env = FakeClothBackend()
    obs = env.observe()
    a = select_action(
        obs, env.config, policy=policy, primitive="fling", rng=np.random.default_rng(3)
    )
    b = select_action(
        obs, env.config, policy=policy, primitive="fling", rng=np.random.default_rng(3)
    )
    assert a == b and a.primitive == "fling"
    assert a.pixels[0] != a.pixels[1]
    mask = foreground_mask(obs)
    assert all(mask[int(v), int(u)] for u, v in a.pixels)
    world = pixels_to_world(obs, np.array(a.pixels))
    assert (
        env.config.min_grasp_separation_m <= np.linalg.norm(world[1] - world[0]) <= 0.3
    )
    assert (np.abs(world[:, :2]) <= 0.45).all()
    if policy == "geometry_guided":
        assert "privileged_depth" in a.policy


def test_uniform_rng_changes_choices_not_just_widest_pair():
    obs = observation()
    pairs = {
        select_action(obs, SceneConfig(), rng=np.random.default_rng(i)).pixels
        for i in range(10)
    }
    assert len(pairs) > 5


def test_geometry_widest_boundary_and_tie_break():
    env = FakeClothBackend()
    obs = env.observe()
    candidates = feasible_candidates(obs, env.config)
    distances = [
        np.linalg.norm(np.diff(pixels_to_world(obs, np.array(a.pixels)), axis=0))
        for a in candidates
    ]
    assert all(a >= b for a, b in zip(distances, distances[1:]))
    assert select_action(obs, env.config, policy="geometry_guided") == candidates[0]
    assert candidates == feasible_candidates(obs, env.config)


def test_depth_holes_workspace_and_no_valid_pair():
    obs = observation()
    obs.depth[:] = np.nan
    with pytest.raises(NoValidPairError):
        select_action(obs, SceneConfig())
    obs = observation()
    with pytest.raises(NoValidPairError):
        select_action(
            obs,
            SelectionConfig(
                grasp_workspace_m=0.3,
                min_grasp_separation_m=0.28,
                max_grasp_separation_m=0.29,
                max_attempts=10,
            ),
        )
    obs.rgb[:] = [64, 71, 77]
    with pytest.raises(NoValidPairError):
        select_action(obs, SceneConfig(), policy="geometry_guided")


def test_missing_depth_no_oracle_fallback():
    obs = observation()
    obs.depth = None
    with pytest.raises(NoValidPairError, match="depth"):
        select_action(obs, SceneConfig())


def test_search_budgets_and_bad_inputs():
    with pytest.raises(ValueError):
        select_action(observation(), SelectionConfig(max_attempts=0))
    with pytest.raises(ValueError):
        select_action(observation(), SceneConfig(), policy="oracle")
    with pytest.raises(ValueError):
        pixels_to_world(observation(), np.array([[np.nan, 3]]))
    with pytest.raises(ValueError):
        pixels_to_world(observation(), np.array([[-1, 3]]))
    obs = observation()
    obs.calibration = {}
    with pytest.raises(ValueError, match="calibration"):
        select_action(obs, SceneConfig())


def test_no_hidden_particle_state_or_simulator_dependency():
    # Only the observation and config are passed; selection has no backend access.
    env = FakeClothBackend()
    obs = env.observe()
    a = select_action(obs, env.config)
    env.positions[:] += 100
    assert select_action(obs, env.config) == a


def test_bounded_boundary_subsample():
    config = replace(SelectionConfig(), max_boundary_candidates=8)
    assert len(feasible_candidates(observation(), config, max_pairs=100)) <= 8 * 7 // 2
