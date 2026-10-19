import json
from dataclasses import replace

import numpy as np
import pytest
from test_primitives_fake import FakeClothBackend

from cloth_fling.contracts import GraspAction
from cloth_fling.diagnostics import run_diagnostic
from cloth_fling.selection import select_action
from cloth_fling.sim.primitives import PrimitiveConfig, execute_action


@pytest.fixture
def short_config():
    return PrimitiveConfig(
        lift_height_m=0.05,
        lift_seconds=0.5,
        tauten_seconds=0.3,
        laydown_seconds=0.6,
        drag_seconds=0.3,
        fling_segment_seconds=0.2,
        laydown_distance_m=0.025,
        drag_distance_m=0.015,
        swing_distance_m=0.035,
        pullback_distance_m=0.025,
        place_distance_m=0.01,
        max_strain=0.3,
        settle_hold_seconds=0.05,
        settle_timeout_seconds=2,
    )


def action(env, primitive="laydown"):
    return select_action(
        env.observe(), env.config, policy="geometry_guided", primitive=primitive
    )


@pytest.mark.parametrize(
    "primitive,phases",
    [
        (
            "laydown",
            [
                "attach",
                "lift",
                "tauten",
                "diagonal_laydown",
                "drag",
                "release",
                "settle",
            ],
        ),
        (
            "fling",
            [
                "attach",
                "lift",
                "tauten",
                "swing",
                "pullback",
                "lower",
                "place",
                "release",
                "settle",
            ],
        ),
    ],
)
def test_lifecycle_retained_and_measured(short_config, primitive, phases):
    env = FakeClothBackend()
    result = execute_action(env, action(env, primitive), short_config)
    assert result.execution_valid, (result.reason, result.diagnostics)
    assert result.evidence["phases"] == phases
    assert env.reset_calls == 0 and env.attach_calls == 1 and env.release_calls == 1
    assert env.attached_ids == () and result.diagnostics["settled"]
    assert result.after.observation_id != result.before.observation_id
    assert result.coverage_after == env.coverage()[0]
    assert (
        result.diagnostics["coverage_gain"]
        == result.coverage_after - result.coverage_before
    )
    assert all(len(ids) == 2 for _, _, ids in env.commands)
    assert env.release_records[0][2][list(env.release_records[0][1]), 2].max() <= 0.02
    assert result.diagnostics["strain_bound_available"]
    traces = result.evidence["phase_traces"]
    assert all("cloth_positions_m" in t and "rms_speed_m_s" in t for t in traces)
    assert result.diagnostics["behavior_passed"] is None
    commands = np.array([target for _, target, _ in env.commands])
    # Include start at rest and final endpoint to test actual discrete command bounds.
    velocity = np.diff(commands, axis=0) / env.config.sim_dt
    acceleration = np.diff(velocity, axis=0) / env.config.sim_dt
    assert np.linalg.norm(velocity, axis=2).max() <= short_config.max_speed_m_s + 1e-6
    assert (
        np.linalg.norm(acceleration, axis=2).max()
        <= short_config.max_acceleration_m_s2 + 1e-6
    )
    # No hidden stops at swing/pullback/lower or laydown/drag internal knots.
    for phase in ["pullback", "lower", "place"] if primitive == "fling" else ["drag"]:
        trace = next(t for t in traces if t["phase"] == phase)
        times = np.array([t for t, _, _ in env.commands])
        index = np.argmin(abs(times - trace["simulated_seconds"]))
        assert np.linalg.norm(velocity[max(0, index - 1)]) > 1e-5


def test_negative_gain_and_determinism(short_config):
    envs = [FakeClothBackend(), FakeClothBackend()]
    outcomes = [execute_action(env, action(env), short_config) for env in envs]
    assert outcomes[0].execution_valid, outcomes[0].reason
    assert outcomes[0].diagnostics["coverage_gain"] < 0
    np.testing.assert_equal(outcomes[0].evidence, outcomes[1].evidence)
    np.testing.assert_array_equal(envs[0].positions, envs[1].positions)


@pytest.mark.parametrize(
    "limit,reason",
    [
        ("max_simulated_seconds", "simulated_deadline"),
        ("max_wall_seconds", "wall_deadline"),
    ],
)
def test_deadline_releases(short_config, limit, reason):
    env = FakeClothBackend()
    c = replace(
        short_config, **{limit: 0.025 if limit == "max_simulated_seconds" else 1e-9}
    )
    result = execute_action(env, action(env), c)
    assert not result.execution_valid and result.truncated
    assert reason in result.reason
    assert env.attached_ids == () and env.release_calls == 1 and env.reset_calls == 0
    assert result.simulated_seconds <= c.max_simulated_seconds


@pytest.mark.parametrize("failure", ["step", "attach", "targets", "nan", "release"])
def test_unexpected_and_nonfinite_errors_explicit_and_cleanup(short_config, failure):
    env = FakeClothBackend()
    selected = action(env)
    if failure == "step":
        env.step = lambda *args: (_ for _ in ()).throw(RuntimeError("broken step"))
    elif failure == "attach":

        def broken_attach(pixels):
            env.attached_ids = (0, 1)
            raise RuntimeError("partial attach")

        env.attach = broken_attach
    elif failure == "targets":
        env.set_attachment_targets = lambda *args: (_ for _ in ()).throw(
            RuntimeError("target error")
        )
    elif failure == "nan":
        step = env.step

        def bad_step():
            step()
            env.positions[3, 0] = np.nan

        env.step = bad_step
    elif failure == "release":
        release = env.release

        def bad_release():
            release()
            raise RuntimeError("release backend error")

        env.release = bad_release
    result = execute_action(env, selected, short_config)
    assert not result.execution_valid
    assert result.reason != "completed" and result.diagnostics["errors"]
    assert env.release_calls >= 1 and env.attached_ids == ()
    if failure == "nan":
        assert not result.diagnostics["terminal_observation_valid"]
    else:
        assert result.after is not None


def test_invalid_action_does_not_step_or_reset(short_config):
    env = FakeClothBackend()
    result = execute_action(env, GraspAction(((0, 0), (63, 63))), short_config)
    assert not result.execution_valid and env.step_calls == 0
    assert (
        "foreground" in result.reason
        and env.reset_calls == 0
        and env.release_calls == 1
    )


def test_release_retains_momentum():
    env = FakeClothBackend()
    env.attach(action(env).pixels)
    target = env.positions[list(env.attached_ids)].copy()
    target[:, 2] += 0.001
    env.set_attachment_targets(target)
    env.step()
    before = env.get_velocities()
    assert np.linalg.norm(before) > 0
    env.release()
    np.testing.assert_array_equal(env.get_velocities(), before)


def test_settle_requires_continuous_window(short_config):
    env = FakeClothBackend()
    step = env.step
    released_steps = 0

    def pulse():
        nonlocal released_steps
        step()
        if not env.attached_ids:
            released_steps += 1
            if released_steps <= 15 and released_steps % 4 == 0:
                env.velocities[:, 0] = 0.02
            else:
                env.velocities[:] = 0

    env.step = pulse
    result = execute_action(env, action(env), short_config)
    assert result.execution_valid, result.reason
    assert released_steps >= 17  # last pulse at 12; must wait five quiet steps


def test_settle_timeout_is_invalid(short_config):
    env = FakeClothBackend()
    step = env.step

    def never_quiet():
        step()
        if not env.attached_ids:
            env.velocities[:, 0] = 0.02

    env.step = never_quiet
    result = execute_action(
        env, action(env), replace(short_config, settle_timeout_seconds=0.1)
    )
    assert not result.execution_valid and result.reason == "settle_timeout"
    assert not result.diagnostics["settled"]


def test_diagnostic_fixed_pair_restores_fixture(short_config, tmp_path):
    env = FakeClothBackend()
    positions, velocities, clock = (
        env.positions.copy(),
        env.velocities.copy(),
        env.simulated_time,
    )
    report = run_diagnostic(
        env, seed=4, output_dir=tmp_path, primitive_config=short_config
    )
    assert report["execution_valid"], report
    assert report["final_restore_valid"] and all(
        x["valid"] for x in report["restore_checks"]
    )
    assert len(report["runs"]) == 2
    assert report["runs"][0]["pixels"] == report["runs"][1]["pixels"]
    assert report["runs"][0]["coverage_before"] == report["runs"][1]["coverage_before"]
    assert not report["quality_qualified"] and env.reset_calls == 0
    np.testing.assert_array_equal(env.positions, positions)
    np.testing.assert_array_equal(env.velocities, velocities)
    assert env.simulated_time == clock
    assert json.loads((tmp_path / "diagnostic.json").read_text()) == report
    json.dumps(report, allow_nan=False)


def test_diagnostic_rejects_broken_restore(short_config):
    env = FakeClothBackend()
    restore = env.restore

    def broken_restore(state):
        restore(state)
        env.positions[0, 0] += 0.01
        return env.observe()

    env.restore = broken_restore
    report = run_diagnostic(env, primitive_config=short_config)
    assert not report["execution_valid"] and not report["final_restore_valid"]
    assert not report["runs"] and report["errors"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_simulated_seconds": 31},
        {"max_wall_seconds": 121},
        {"settle_timeout_seconds": 11},
        {"settle_rms_m_s": 0.02},
        {"max_speed_m_s": float("nan")},
    ],
)
def test_config_hard_bounds(kwargs):
    with pytest.raises(ValueError):
        PrimitiveConfig(**kwargs)


def test_compressed_fixture_closed_loop_tauten(short_config):
    env = FakeClothBackend()
    env.positions[:, :2] *= 0.8
    selected = action(env)
    result = execute_action(env, selected, short_config)
    assert result.execution_valid, (result.reason, result.diagnostics)
    trace = [t for t in result.evidence["phase_traces"] if t["phase"] == "tauten"]
    spans = [
        np.linalg.norm(np.diff(np.asarray(t["commanded_grasp_positions_m"]), axis=0))
        for t in trace
    ]
    assert spans[-1] > spans[0] + 0.005
    cap = result.diagnostics["rest_geodesic_distance_m"] * (1 + short_config.max_strain)
    assert max(spans) <= cap + 1e-9


def test_rest_geometry_missing_is_labeled_fallback(short_config):
    env = FakeClothBackend()

    # Hide optional metadata behind the required backend interface.
    class RequiredOnly:
        def __getattr__(self, name):
            if name in {"rest_positions", "edges", "faces"}:
                raise AttributeError(name)
            return getattr(env, name)

        def snapshot(self):
            return {
                k: v
                for k, v in env.snapshot().items()
                if k not in {"rest_positions", "edges"}
            }

    result = execute_action(RequiredOnly(), action(env), short_config)
    assert result.execution_valid, result.reason
    assert not result.diagnostics["strain_bound_available"]
    assert result.diagnostics["rest_geodesic_distance_m"] is None
    assert (
        "no material strain guarantee"
        in result.diagnostics["proxy_labels"]["missing_rest"]
    )


def test_strain_limit_fails_explicitly_and_releases(short_config):
    env = FakeClothBackend()
    step = env.step

    def overstrain():
        step()
        env.positions[12, 0] += 0.1

    env.step = overstrain
    result = execute_action(env, action(env), short_config)
    assert not result.execution_valid and result.reason == "cloth_strain_limit"
    assert not env.attached_ids and env.release_calls == 1


def test_release_velocity_mutation_is_invalid(short_config):
    env = FakeClothBackend()
    release = env.release

    def zero_momentum():
        release()
        env.velocities[:] = 0
        env.velocities[0, 0] = (
            0.1  # detectable violation even with a quiet final command
        )

    env.release = zero_momentum
    result = execute_action(env, action(env), short_config)
    assert not result.execution_valid and result.reason == "release_changed_velocity"
    assert result.diagnostics["release_velocity_delta_max_m_s"] >= 0.09


def test_valid_actions_continue_without_reset(short_config):
    env = FakeClothBackend()
    first = execute_action(env, action(env), short_config)
    assert first.execution_valid
    second = execute_action(env, action(env), short_config)
    assert second.execution_valid, second.reason
    assert second.before.observation_id == first.after.observation_id
    np.testing.assert_array_equal(second.before.rgb, first.after.rgb)
    assert second.coverage_before == first.coverage_after
    assert env.reset_calls == 0 and env.attach_calls == 2


def test_manual_diagnostic_pair_artifacts(short_config, tmp_path):
    from PIL import Image

    env = FakeClothBackend()
    selected = action(env)
    report = run_diagnostic(
        env, action=selected, primitive_config=short_config, output_dir=tmp_path
    )
    assert report["execution_valid"], report["errors"]
    assert report["pair_selection"].startswith("manual_fixed_pair")
    assert report["fixed_pixels"] == [list(pixel) for pixel in selected.pixels]
    for run in report["runs"]:
        refs = run["artifacts"]
        with Image.open(tmp_path / refs["before_rgb"]) as image:
            assert image.size == (64, 64)
        with np.load(tmp_path / refs["depth"], allow_pickle=False) as depths:
            assert depths["before"].shape == (64, 64)
        assert (
            json.loads((tmp_path / refs["calibration"]).read_text())["before"][
                "pixel_center_offset"
            ]
            == 0.5
        )
    with np.load(
        tmp_path / report["fixture_artifacts"]["arrays"], allow_pickle=False
    ) as state:
        assert len(state.files) > 0


def test_default_laydown_blend_stays_above_table():
    from cloth_fling.sim.primitives import _bezier, _trajectory

    c = PrimitiveConfig()
    start = np.array([[-0.1, 0, c.lift_height_m], [0.1, 0, c.lift_height_m]])
    laid = start + [0, c.laydown_distance_m, 0]
    laid[:, 2] = c.release_height_m
    dragged = laid + [0, c.drag_distance_m, 0]
    controls, _ = _trajectory(
        [start, laid, dragged],
        [c.laydown_seconds, c.drag_seconds],
        c.max_speed_m_s,
        c.max_acceleration_m_s2,
    )
    for control in controls:
        for u in np.linspace(0, 1, 100):
            assert _bezier(control, u)[:, 2].min() >= c.release_height_m - 1e-12
    # Vertical speed becomes zero at the join while horizontal motion continues.
    join_velocity = 5 * (controls[1][1] - controls[1][0]) / c.drag_seconds
    assert (join_velocity[:, 1] > 0).all() and (join_velocity[:, 2] == 0).all()
