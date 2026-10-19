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
    return select_action(env.observe(), env.config, policy="geometry_guided", primitive=primitive)


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
    assert result.diagnostics["coverage_gain"] == result.coverage_after - result.coverage_before
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
    assert np.linalg.norm(acceleration, axis=2).max() <= short_config.max_acceleration_m_s2 + 1e-6
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
    c = replace(short_config, **{limit: 0.025 if limit == "max_simulated_seconds" else 1e-9})
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
    assert "foreground" in result.reason and env.reset_calls == 0 and env.release_calls == 1


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
    result = execute_action(env, action(env), replace(short_config, settle_timeout_seconds=0.1))
    assert not result.execution_valid and result.reason == "settle_timeout"
    assert not result.diagnostics["settled"]


def test_diagnostic_fixed_pair_restores_fixture(short_config, tmp_path):
    env = FakeClothBackend()
    positions, velocities, clock = (
        env.positions.copy(),
        env.velocities.copy(),
        env.simulated_time,
    )
    report = run_diagnostic(env, seed=4, output_dir=tmp_path, primitive_config=short_config)
    assert report["execution_valid"], report
    assert report["final_restore_valid"] and all(x["valid"] for x in report["restore_checks"])
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
        np.linalg.norm(np.diff(np.asarray(t["commanded_grasp_positions_m"]), axis=0)) for t in trace
    ]
    assert spans[-1] > spans[0] + 0.005
    cap = result.diagnostics["rest_flat_chord_m"] * (1 + short_config.max_strain)
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
            return {k: v for k, v in env.snapshot().items() if k not in {"rest_positions", "edges"}}

    result = execute_action(RequiredOnly(), action(env), short_config)
    assert result.execution_valid, result.reason
    assert not result.diagnostics["strain_bound_available"]
    assert result.diagnostics["rest_geodesic_distance_m"] is None
    assert result.diagnostics["rest_flat_chord_m"] is None
    assert result.diagnostics["graph_geodesic_distance"] is None
    assert result.diagnostics["tauten"]["material_bound"] == "fallback"
    assert "no material strain guarantee" in result.diagnostics["proxy_labels"]["missing_rest"]


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
        env.velocities[0, 0] = 0.1  # detectable violation even with a quiet final command

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
    with np.load(tmp_path / report["fixture_artifacts"]["arrays"], allow_pickle=False) as state:
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


class CollectionTimeout(TimeoutError):
    """Collector-owned cancellation without importing the collection implementation."""


@pytest.mark.parametrize(
    "stage",
    [
        "initial_observe",
        "attach",
        "targets",
        "step",
        "terminal_observe",
        "terminal_state",
        "terminal_coverage",
    ],
)
def test_collector_timeouts_propagate_after_release_cleanup(short_config, stage):
    env = FakeClothBackend()
    selected = action(env)
    timeout = CollectionTimeout(f"collector cancelled at {stage}")

    def fail(*args, **kwargs):
        raise timeout

    if stage == "initial_observe":
        env.observe = fail
    elif stage == "attach":

        def partial_attach(*args):
            env.attached_ids = (0, 1)
            raise timeout

        env.attach = partial_attach
    elif stage == "targets":
        env.set_attachment_targets = fail
    elif stage == "step":
        env.step = fail
    else:
        method = {
            "terminal_observe": "observe",
            "terminal_state": "get_positions",
            "terminal_coverage": "coverage",
        }[stage]
        original = getattr(env, method)

        def terminal_failure(*args, **kwargs):
            if env.release_calls:
                assert not env.attached_ids
                raise timeout
            return original(*args, **kwargs)

        setattr(env, method, terminal_failure)
        # Failure before motion routes state/coverage timeouts specifically through
        # terminal capture, rather than the post-release settling samples.
        if stage != "terminal_observe":
            selected = GraspAction(((0, 0), (63, 63)))
    with pytest.raises(CollectionTimeout) as caught:
        execute_action(env, selected, short_config)
    assert caught.value is timeout
    assert env.release_calls == 1 and not env.attached_ids and env.reset_calls == 0


def test_cleanup_timeout_is_not_swallowed(short_config):
    env = FakeClothBackend()
    selected = GraspAction(((0, 0), (63, 63)))
    release = env.release
    timeout = CollectionTimeout("cleanup deadline")

    def slow_release():
        release()
        raise timeout

    env.release = slow_release
    with pytest.raises(CollectionTimeout) as caught:
        execute_action(env, selected, short_config)
    assert caught.value is timeout and not env.attached_ids


@pytest.mark.parametrize("missing", ["raises", "returns_none"])
def test_no_fabricated_terminal_observation(short_config, missing):
    env = FakeClothBackend()
    selected = action(env)
    observe = env.observe

    def unavailable_terminal():
        if env.release_calls:
            if missing == "raises":
                raise RuntimeError("terminal camera unavailable")
            return None
        return observe()

    env.observe = unavailable_terminal
    result = execute_action(env, selected, short_config)
    assert not result.execution_valid and result.after is None
    assert result.before is not None
    assert not result.diagnostics["terminal_observation_valid"]
    assert result.reason.startswith("terminal_capture_failed")
    assert np.isnan(result.coverage_after)
    assert env.release_calls == 1 and not env.attached_ids


def test_terminal_measurement_failure_keeps_actual_captured_observation(short_config):
    env = FakeClothBackend()
    coverage = env.coverage

    def failed_terminal_coverage():
        if env.release_calls:
            raise RuntimeError("terminal coverage unavailable")
        return coverage()

    env.coverage = failed_terminal_coverage
    result = execute_action(env, GraspAction(((0, 0), (63, 63))), short_config)
    assert not result.execution_valid
    assert result.after is not None and result.after is not result.before
    np.testing.assert_array_equal(result.after.rgb, env.observe().rgb)
    assert not result.diagnostics["terminal_observation_valid"]


def test_diagnostic_missing_after_has_no_fabricated_artifact(short_config, tmp_path):
    env = FakeClothBackend()
    observe = env.observe

    def missing_after():
        if env.release_calls and env.step_calls and not env.attached_ids:
            raise RuntimeError("terminal camera unavailable")
        return observe()

    env.observe = missing_after
    report = run_diagnostic(env, primitive_config=short_config, output_dir=tmp_path)
    assert not report["execution_valid"] and report["runs"]
    run = report["runs"][0]
    assert not run["diagnostics"]["terminal_observation_valid"]
    assert run["artifacts"]["after_rgb"] is None
    assert not (tmp_path / "laydown_after.png").exists()
    with np.load(tmp_path / run["artifacts"]["depth"], allow_pickle=False) as depth:
        assert "before" in depth and "after" not in depth


def test_diagnostic_propagates_timeout_after_fixture_restore(short_config):
    env = FakeClothBackend()
    positions = env.positions.copy()
    timeout = CollectionTimeout("diagnostic cancelled")

    def timed_out_step():
        raise timeout

    env.step = timed_out_step
    with pytest.raises(CollectionTimeout) as caught:
        run_diagnostic(env, primitive_config=short_config)
    assert caught.value is timeout
    assert env.release_calls == 1 and not env.attached_ids
    np.testing.assert_array_equal(env.positions, positions)
    assert env.simulated_time == 0


def test_triangulated_graph_bias_does_not_set_material_span(short_config):
    env = FakeClothBackend()
    env.config = replace(env.config, max_grasp_separation_m=0.55)
    env.positions[:, :2] *= 0.8
    # The existing asymmetric triangle diagonals force the selected pair's graph
    # path around a corner. Keep the production 0.10 ceiling for this regression.
    config = replace(short_config, max_strain=PrimitiveConfig().max_strain)
    first = execute_action(env, action(env), config)
    assert first.execution_valid, (first.reason, first.diagnostics)
    diagnostics = first.diagnostics
    graph, chord = (
        diagnostics["graph_geodesic_distance"],
        diagnostics["rest_flat_chord_m"],
    )
    assert graph > 1.3 * chord
    assert diagnostics["tauten"]["span_cap_m"] == pytest.approx(
        min(env.config.max_grasp_separation_m, chord * (1 + config.max_strain))
    )
    assert diagnostics["rest_geodesic_distance_m"] == graph  # historical field stays graph-only
    assert diagnostics["tauten"]["target_span_m"] == pytest.approx(chord * config.taut_rest_ratio)
    spans = [
        np.linalg.norm(np.diff(t["commanded_grasp_positions_m"], axis=0))
        for t in first.evidence["phase_traces"]
        if t["phase"] == "tauten"
    ]
    assert max(spans) <= chord * config.taut_rest_ratio + 1e-9
    assert diagnostics["max_edge_strain_proxy"] <= config.max_strain
    second = execute_action(env, action(env), config)
    assert second.execution_valid, second.reason
    assert second.before.observation_id == first.after.observation_id
    assert second.coverage_before == first.coverage_after
    assert env.reset_calls == 0


def _tauten_fixture(config):
    import time

    from cloth_fling.sim.primitives import _Execution, _rest_geometry, _rest_path

    env = FakeClothBackend()
    env.positions[:, :2] *= 0.8
    env.positions[:, 2] = config.lift_height_m
    env.attached_ids = (0, 24)
    env.targets = env.positions[list(env.attached_ids)].copy()
    run = _Execution(env, config, time.monotonic(), env.simulated_time)
    run.ids = run.grasp_ids = env.attached_ids
    run.target = env.targets.copy()
    run.geometry = _rest_geometry(env, len(env.positions))
    run.path, run.rest_distance = _rest_path(run.geometry, run.ids, len(env.positions))
    run.rest_flat_chord = float(np.linalg.norm(env.rest_positions[24] - env.rest_positions[0]))
    return env, run


def test_local_strain_guard_brakes_moving_span_before_hard_ceiling(short_config):
    config = replace(short_config, max_strain=0.1, max_acceleration_m_s2=0.2, tauten_seconds=2)
    env, run = _tauten_fixture(config)
    # Relax the compressed setup with held attachments before injecting a local
    # disturbance; initial solver transients have their own predictive-stop test.
    for _ in range(20):
        env.step()
    run.sim_started = env.simulated_time
    step = env.step
    previous_span = np.linalg.norm(env.targets[1] - env.targets[0])
    disturbed = False

    def localized_extension():
        nonlocal previous_span, disturbed
        step()
        span = np.linalg.norm(env.targets[1] - env.targets[0])
        span_rate = (span - previous_span) / env.config.sim_dt
        previous_span = span
        if disturbed or span_rate > 0.02:
            # A local cloth extension disturbance below the hard ceiling. The
            # unconstrained particles respond inside the fake, never in the primitive.
            disturbed = True
            env.positions[12] = env.positions[11] + [0.05 * 1.085, 0, 0]

    env.step = localized_extension
    run.tauten()
    assert disturbed
    diagnostics = run.tauten_diagnostics
    assert diagnostics["stop_reason"] == "local_strain_guard"
    brake = diagnostics["strain_brake"]
    assert brake["span_rate_m_s"] > 0.02
    assert brake["max_edge_strain_proxy"] == pytest.approx(0.085)
    assert brake["max_edge_strain_proxy"] < config.max_strain
    assert brake["predicted_edge_strain_proxy"] >= diagnostics["soft_strain_limit"]
    assert run.max_strain < config.max_strain
    assert run.command_acceleration <= config.max_acceleration_m_s2 + 1e-6
    np.testing.assert_allclose(run.previous_command_velocity, 0, atol=1e-12)
    assert diagnostics["final_span_m"] < diagnostics["target_span_m"]
    assert env.simulated_time < config.tauten_seconds  # transition after braking, no long hold


def test_positive_local_strain_rate_triggers_predictive_stop(short_config):
    config = replace(short_config, max_strain=0.1)
    env, run = _tauten_fixture(config)
    env.positions[12] = env.positions[11] + [0.05 * 1.07, 0, 0]
    env.velocities[12, 0] = 0.4  # measured outward velocity predicts the local extension
    run.tauten()
    diagnostics = run.tauten_diagnostics
    assert diagnostics["stop_reason"] == "local_strain_guard"
    assert diagnostics["strain_brake"]["max_edge_strain_proxy"] < diagnostics["soft_strain_limit"]
    assert (
        diagnostics["strain_brake"]["predicted_edge_strain_proxy"]
        >= diagnostics["soft_strain_limit"]
    )
    assert diagnostics["final_span_m"] == pytest.approx(diagnostics["initial_span_m"])
    assert run.max_strain < config.max_strain
