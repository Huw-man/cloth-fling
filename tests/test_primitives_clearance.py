"""Clearance lifecycle guards with CPU fakes; no native physics qualification."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from test_primitives_fake import FakeClothBackend

from cloth_fling.selection import select_action
from cloth_fling.sim.primitives import PrimitiveConfig, execute_action


class HangingClothFake(FakeClothBackend):
    """Scripted hanging length isolates surface guards from a fake solver's tuning.

    All free-particle motion here is test-backend behavior, never primitive control.
    No rest metadata means these tests make no material-strain claim.
    """

    particle_diameter = 0.01

    def __init__(self, hanging_length=0.34, *, stays_supported=False):
        super().__init__(metadata=False)
        del self.rest_positions, self.edges, self.lengths
        self.hanging_length = hanging_length
        self.stays_supported = stays_supported
        self.phase_hint = None
        self.disturb = None
        self.physics_history = []

    def snapshot(self):
        return {}

    def step(self, count=1):
        for _ in range(count):
            previous = self.positions.copy()
            free = np.ones(len(self.positions), dtype=bool)
            free[list(self.attached_ids)] = False
            floor = self.particle_diameter / 2
            height = floor
            if self.attached_ids:
                self.positions[list(self.attached_ids)] = self.targets
                if not self.stays_supported:
                    height = max(floor, self.targets[:, 2].min() - self.hanging_length)
            self.positions[free, 2] = height
            if self.disturb is not None:
                self.disturb(self, free)
            self.velocities = (self.positions - previous) / self.config.sim_dt
            self.simulated_time += self.config.sim_dt
            self.step_calls += 1
            self.physics_history.append(
                (self.phase_hint, self.simulated_time, self.positions.copy())
            )


@pytest.fixture
def clearance_config():
    # Keep the production 0.30 m initial lift and 0.05 m increments. Shorten
    # placement/settling only; clearance and action guards remain unchanged.
    return PrimitiveConfig(
        tauten_seconds=0.2,
        laydown_seconds=0.6,
        drag_seconds=0.3,
        laydown_distance_m=0.025,
        drag_distance_m=0.015,
        swing_distance_m=0.035,
        pullback_distance_m=0.025,
        place_distance_m=0.01,
        settle_hold_seconds=0.05,
        settle_timeout_seconds=2,
    )


def selected(env, primitive="laydown"):
    return select_action(env.observe(), env.config, policy="geometry_guided", primitive=primitive)


def phase_observer(env, trace):
    env.phase_hint = trace["phase"]


@pytest.mark.parametrize("primitive", ["laydown", "fling"])
def test_raised_grasps_do_not_prove_whole_cloth_clearance(clearance_config, primitive):
    env = HangingClothFake(stays_supported=True)
    c = replace(clearance_config, clearance_max_height_m=0.40)
    result = execute_action(env, selected(env, primitive), c)
    assert not result.execution_valid and result.reason == "lift_max_height"
    d = result.diagnostics["clearance"]
    assert not d["gate_passed"] and d["failure_reason"] == "lift_max_height"
    assert d["achieved_height_m"] == pytest.approx(0.40)
    assert d["measured_min_clearance_m"] == pytest.approx(0)
    assert d["hold_seconds"] == 0 and d["near_table_fraction"] > 0.9
    assert "swing" not in result.evidence["phases"]
    assert "diagonal_laydown" not in result.evidence["phases"]
    assert max(t[:, 2].max() for _, t, _ in env.commands) <= 0.40 + 1e-12
    assert env.release_calls == 1 and not env.attached_ids and env.reset_calls == 0
    assert result.after is not None
    np.testing.assert_array_equal(env.velocities, env.release_records[0][3])


def test_successful_incremental_clearance_after_initial_lift_and_tauten(clearance_config):
    env = HangingClothFake()
    result = execute_action(env, selected(env), clearance_config, trace_callback=phase_observer)
    assert result.execution_valid, result.reason
    phases = result.evidence["phases"]
    assert phases[:5] == ["attach", "lift", "tauten", "clearance", "diagonal_laydown"]
    traces = result.evidence["phase_traces"]
    lifted = [t for t in traces if t["phase"] == "lift"][-1]
    assert np.array(lifted["commanded_grasp_positions_m"])[:, 2] == pytest.approx([0.30, 0.30])
    taut = [t for t in traces if t["phase"] == "tauten"][-1]
    assert taut["min_surface_clearance_m"] < clearance_config.clearance_m
    heights = [target[0, 2] for _, target, _ in env.commands]
    assert any(abs(z - 0.35) < 1e-10 for z in heights)
    assert max(heights) == pytest.approx(0.40)
    d = result.diagnostics["clearance"]
    assert d["gate_passed"] and d["failure_reason"] is None
    assert d["achieved_height_m"] == pytest.approx(0.40)
    assert d["hold_seconds"] >= clearance_config.clearance_hold_seconds - 1e-9
    assert d["hold_min_clearance_m"] >= clearance_config.clearance_m
    commands = np.array([target for _, target, _ in env.commands])
    velocity = np.diff(commands, axis=0) / env.config.sim_dt
    acceleration = np.diff(velocity, axis=0) / env.config.sim_dt
    assert np.linalg.norm(velocity, axis=2).max() <= clearance_config.max_speed_m_s + 1e-6
    assert (
        np.linalg.norm(acceleration, axis=2).max() <= clearance_config.max_acceleration_m_s2 + 1e-6
    )
    assert env.release_calls == 1 and env.reset_calls == 0


@pytest.mark.parametrize("phase", ["swing", "pullback"])
def test_one_tick_clearance_loss_between_trace_samples_aborts(clearance_config, phase):
    env = HangingClothFake(hanging_length=0.28)
    injected = []

    def contact(backend, free):
        if backend.phase_hint == phase and not injected:
            backend.positions[np.flatnonzero(free)[0], 2] = backend.particle_diameter / 2
            injected.append(backend.simulated_time + backend.config.sim_dt)

    env.disturb = contact
    c = replace(clearance_config, trace_interval_seconds=30)
    result = execute_action(env, selected(env, "fling"), c, trace_callback=phase_observer)
    assert not result.execution_valid and result.reason == "premature_table_contact"
    d = result.diagnostics["clearance"]
    assert d["gate_passed"] and d["hold_min_clearance_m"] >= c.clearance_m
    assert d["protected_min_clearance_m"] == pytest.approx(0)
    assert d["failure_reason"] == "premature_table_contact"
    assert result.simulated_seconds == pytest.approx(injected[0])
    assert "lower" not in result.evidence["phases"]
    # A failing tick is captured even when ordinary traces are heavily decimated.
    failed = result.evidence["phase_traces"][-1]
    assert failed["phase"] == phase and failed["min_surface_clearance_m"] == pytest.approx(0)
    assert env.release_calls == 1 and not env.attached_ids and env.reset_calls == 0
    np.testing.assert_array_equal(env.velocities, env.release_records[0][3])


def test_surface_guard_rejects_centres_that_clear_but_surfaces_do_not(clearance_config):
    env = HangingClothFake(hanging_length=0.277)
    # At .30 m, free centres are .023 m above table, but surfaces are only .018 m.
    result = execute_action(env, selected(env), clearance_config)
    assert result.execution_valid, result.reason
    d = result.diagnostics["clearance"]
    assert d["measured_min_clearance_m"] == pytest.approx(0.018)
    assert d["achieved_height_m"] == pytest.approx(0.35)


def test_hold_restarts_after_single_low_tick(clearance_config):
    env = HangingClothFake(hanging_length=0.28)
    clearance_times = []
    disturbed = []

    def temporary_contact(backend, free):
        if backend.phase_hint == "clearance" and backend.targets[0, 2] > 0.349:
            clearance_times.append(backend.simulated_time + backend.config.sim_dt)
            if len(clearance_times) == 6:
                backend.positions[free, 2] = 0.024  # .019 m surface; resets hold
                disturbed.append(clearance_times[-1])

    env.disturb = temporary_contact
    result = execute_action(env, selected(env), clearance_config, trace_callback=phase_observer)
    assert result.execution_valid, result.reason
    assert disturbed
    last = [t for t in result.evidence["phase_traces"] if t["phase"] == "clearance"][-1]
    assert last["simulated_seconds"] - disturbed[0] >= 0.1
    assert result.diagnostics["clearance"]["hold_seconds"] >= 0.1 - 1e-9


def test_legacy_disabled_is_recorded_without_clearance_gate(clearance_config):
    env = HangingClothFake(stays_supported=True)
    c = replace(clearance_config, clearance_enabled=False)
    result = execute_action(env, selected(env, "fling"), c)
    assert result.execution_valid, result.reason
    assert "clearance" not in result.evidence["phases"]
    assert result.evidence["config"]["clearance_enabled"] is False
    assert result.diagnostics["clearance"]["gate_passed"] is None
    lift = [t for t in result.evidence["phase_traces"] if t["phase"] == "lift"][-1]
    assert np.array(lift["commanded_grasp_positions_m"])[:, 2] == pytest.approx([0.30, 0.30])


def test_clearance_obeys_existing_simulated_deadline(clearance_config):
    env = HangingClothFake(stays_supported=True)
    c = replace(clearance_config, max_simulated_seconds=3)
    result = execute_action(env, selected(env), c)
    assert not result.execution_valid and result.truncated
    assert result.reason == "simulated_deadline"
    assert "clearance" in result.evidence["phases"]
    assert result.diagnostics["clearance"]["failure_reason"] == "simulated_deadline"
    assert result.simulated_seconds <= 3 + 1e-9
    assert env.release_calls == 1 and not env.attached_ids


def test_fling_clearance_protection_ends_at_intended_lower(clearance_config):
    env = HangingClothFake(hanging_length=0.28)
    result = execute_action(env, selected(env, "fling"), clearance_config)
    assert result.execution_valid, result.reason
    assert result.diagnostics["clearance"]["protected_min_clearance_m"] >= 0.020
    lower = [t for t in result.evidence["phase_traces"] if t["phase"] == "lower"]
    assert min(t["min_surface_clearance_m"] for t in lower) < 0.020
    assert result.diagnostics["release_velocity_delta_max_m_s"] == 0
    assert env.release_calls == 1 and env.reset_calls == 0


@pytest.mark.parametrize("enabled", [True, False])
def test_pullback_height_v2_and_exact_legacy_waypoint_heights(
    clearance_config, monkeypatch, enabled
):
    from cloth_fling.sim.primitives import _Execution

    env = HangingClothFake(hanging_length=0.34)
    plans = []
    follow = _Execution.follow

    def capture_plan(run, points, durations, phases):
        if phases == ["swing", "pullback", "lower", "place"]:
            plans.append(np.asarray(points).copy())
        return follow(run, points, durations, phases)

    monkeypatch.setattr(_Execution, "follow", capture_plan)
    c = replace(clearance_config, clearance_enabled=enabled)
    result = execute_action(env, selected(env, "fling"), c, trace_callback=phase_observer)
    assert result.execution_valid, result.reason
    assert len(plans) == 1
    z = plans[0][:, :, 2]
    if enabled:
        assert z[0, 0] == pytest.approx(0.40)
        expected = [z[0, 0], z[0, 0], z[0, 0], 0.075, 0.020]
    else:
        expected = [0.30, 0.30, 0.26, 0.075, 0.020]
    np.testing.assert_array_equal(z, np.broadcast_to(np.array(expected)[:, None], z.shape))
    if enabled:
        # Check actual dense physics-step commands, not just the waypoint endpoints.
        protected = [
            p[list(env.release_records[0][1]), 2]
            for phase, _, p in env.physics_history
            if phase in {"swing", "pullback"}
        ]
        lower = [
            p[list(env.release_records[0][1]), 2]
            for phase, _, p in env.physics_history
            if phase == "lower"
        ]
        assert protected and lower
        np.testing.assert_allclose(protected, 0.40, rtol=0, atol=1e-12)
        assert np.min(lower) < 0.10
        assert result.diagnostics["clearance"]["protected_min_clearance_m"] >= c.clearance_m
    assert result.evidence["primitive_config_version"] == "milestone2_clearance_v2"
    assert env.release_calls == 1 and not env.attached_ids and env.reset_calls == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clearance_enabled": 1},
        {"clearance_increment_m": 0.051},
        {"clearance_increment_m": 0},
        {"clearance_max_height_m": 0.751},
        {"clearance_max_height_m": 0.29},
        {"clearance_m": 0.019},
        {"clearance_hold_seconds": 0.099},
        {"clearance_m": float("nan")},
    ],
)
def test_clearance_config_guards(kwargs):
    with pytest.raises(ValueError):
        PrimitiveConfig(**kwargs)


def test_fallback_particle_radius_is_half_mesh_spacing(clearance_config):
    env = FakeClothBackend()
    result = execute_action(env, selected(env), replace(clearance_config, max_strain=0.3))
    assert result.execution_valid, result.reason
    assert result.diagnostics["clearance"]["particle_radius_m"] == env.config.mesh_spacing_m / 2


def test_backend_diameter_overrides_mesh_spacing(clearance_config):
    env = HangingClothFake(hanging_length=0.277)
    env.particle_diameter = 0.02
    result = execute_action(env, selected(env), clearance_config)
    assert result.execution_valid, result.reason
    d = result.diagnostics["clearance"]
    assert d["particle_radius_m"] == 0.01
    assert d["measured_min_clearance_m"] == pytest.approx(0.013)


def test_strain_violation_during_clearance_aborts_without_relaxing_guard(clearance_config):
    env = FakeClothBackend()
    step = env.step
    env.phase_hint = None

    def overstrain():
        step()
        if env.phase_hint == "clearance":
            env.positions[12, 0] += 0.1

    env.step = overstrain
    c = replace(clearance_config, lift_height_m=0.05, lift_seconds=0.5, max_strain=0.3)
    result = execute_action(env, selected(env), c, trace_callback=phase_observer)
    assert not result.execution_valid and result.reason == "cloth_strain_limit"
    assert result.diagnostics["clearance"]["failure_reason"] == "cloth_strain_limit"
    assert "diagonal_laydown" not in result.evidence["phases"]
    assert env.release_calls == 1 and not env.attached_ids


def test_callback_receives_measured_traces_and_same_backend(clearance_config):
    env = HangingClothFake()
    captured = []

    def capture(backend, trace):
        assert backend is env
        assert "coverage" in trace and "min_surface_clearance_m" in trace
        np.testing.assert_array_equal(trace["cloth_positions_m"], backend.get_positions())
        captured.append(trace)

    result = execute_action(env, selected(env), clearance_config, trace_callback=capture)
    assert result.execution_valid, result.reason
    assert len(captured) == len(result.evidence["phase_traces"])
    assert all(a is b for a, b in zip(captured, result.evidence["phase_traces"], strict=True))


def test_selected_trace_precedes_attachment_and_preserves_before_observation(clearance_config):
    from cloth_fling.selection import pixels_to_world

    env = HangingClothFake()
    observation = env.observe()
    selection = selected(env)
    expected = pixels_to_world(observation, np.array(selection.pixels))
    seen = []

    def observe_selection(backend, trace):
        seen.append(trace["phase"])
        if trace["phase"] == "before":
            assert trace["selected_grasp_positions_m"] is None
        elif trace["phase"] == "selected":
            assert backend.attached_ids == () and backend.attach_calls == 0
            assert trace["attachment_ids"] == []
            assert trace["measured_grasp_positions_m"] == []
            assert trace["commanded_grasp_positions_m"] is None
            np.testing.assert_array_equal(trace["selected_grasp_positions_m"], expected)
        elif trace["phase"] in {"attach", "lift", "tauten", "clearance"}:
            assert len(backend.attached_ids) == 2
            np.testing.assert_array_equal(
                trace["measured_grasp_positions_m"],
                backend.get_positions()[list(backend.attached_ids)],
            )

    result = execute_action(env, selection, clearance_config, trace_callback=observe_selection)
    assert result.execution_valid, result.reason
    assert seen[:3] == ["before", "selected", "attach"]
    np.testing.assert_array_equal(result.before.rgb, observation.rgb)
    np.testing.assert_array_equal(result.before.depth, observation.depth)
    assert result.before.observation_id == observation.observation_id
    assert env.reset_calls == 0


@pytest.mark.parametrize("external_timeout", [False, True])
def test_callback_error_releases_and_timeout_propagates(clearance_config, external_timeout):
    env = HangingClothFake()
    error = (
        TimeoutError("recorder cancelled") if external_timeout else RuntimeError("recorder failed")
    )

    def broken(backend, trace):
        if trace["phase"] == "clearance":
            raise error

    if external_timeout:
        with pytest.raises(TimeoutError) as caught:
            execute_action(env, selected(env), clearance_config, trace_callback=broken)
        assert caught.value is error
    else:
        result = execute_action(env, selected(env), clearance_config, trace_callback=broken)
        assert not result.execution_valid
        assert result.reason == "trace_callback_failed_RuntimeError: recorder failed"
        assert result.after is not None
    assert env.release_calls == 1 and not env.attached_ids and env.reset_calls == 0


@pytest.mark.parametrize("native", [True, False])
def test_contact_accounting_once_per_trace_complete_unchanged_and_noninterfering(
    clearance_config, native
):
    env = HangingClothFake()
    baseline = HangingClothFake()
    calls, callbacks = [], []
    payload = {"schema_version": 1, "profile": "native_coulomb_v1" if native else "legacy"}
    if not native:
        payload["available"] = False

    def accounting(*, reset=False):
        assert reset is False
        if native:
            payload.update(
                epoch=17,
                substeps=2**40 + env.step_calls,
                totals={
                    "normal_impulse_kg_m_s": float(2**40) + 0.25,
                    "tangential_impulse_kg_m_s": env.step_calls * 0.125,
                },
            )
        calls.append(copy.deepcopy(payload))
        return payload  # Reused mutable object must not overwrite earlier evidence.

    def recorder(backend, trace):
        assert backend is env
        assert len(calls) == len(callbacks) + 1
        assert trace["table_contact_accounting"] == calls[-1]
        callbacks.append(trace["table_contact_accounting"])

    env.take_contact_accounting = accounting
    c = replace(clearance_config, trace_interval_seconds=0.4)
    result = execute_action(env, selected(env), c, trace_callback=recorder)
    reference = execute_action(baseline, selected(baseline), c)
    assert result.execution_valid and reference.execution_valid
    traces = result.evidence["phase_traces"]
    assert len(calls) == len(callbacks) == len(traces) < env.step_calls
    assert callbacks == calls
    json.dumps(callbacks, allow_nan=False)
    if native:
        assert callbacks[0]["epoch"] == callbacks[-1]["epoch"] == 17
        assert callbacks[0]["substeps"] == 2**40
        assert callbacks[-1]["substeps"] == 2**40 + env.step_calls
        assert callbacks[0]["totals"]["normal_impulse_kg_m_s"] == float(2**40) + 0.25
    else:
        assert all(
            t == {"schema_version": 1, "profile": "legacy", "available": False} for t in calls
        )
    payload["profile"] = "mutated_after_capture"
    assert all(t["profile"] != payload["profile"] for t in callbacks)
    # Optional measurement must not change commands, cloth state or existing evidence.
    np.testing.assert_equal(env.commands, baseline.commands)
    np.testing.assert_array_equal(env.positions, baseline.positions)
    np.testing.assert_array_equal(env.velocities, baseline.velocities)
    without_contact = [
        {k: v for k, v in t.items() if k != "table_contact_accounting"} for t in traces
    ]
    np.testing.assert_equal(without_contact, reference.evidence["phase_traces"])
    assert env.simulated_time == baseline.simulated_time
    assert env.reset_calls == 0 and env.release_calls == 1


def test_absent_contact_accounting_omits_field_and_records_config_version(clearance_config):
    env = HangingClothFake()
    result = execute_action(env, selected(env), clearance_config)
    assert result.execution_valid, result.reason
    assert all("table_contact_accounting" not in t for t in result.evidence["phase_traces"])
    assert result.evidence["primitive_config_version"] == "milestone2_clearance_v2"
    assert result.evidence["config"]["clearance_enabled"] is True


@pytest.mark.parametrize("external_timeout", [False, True])
def test_contact_accounting_error_fails_capture_and_releases(clearance_config, external_timeout):
    env = HangingClothFake()
    calls, recorded = [], []
    error = (
        TimeoutError("accounting cancelled")
        if external_timeout
        else RuntimeError("nonfinite counters")
    )

    def accounting(*, reset=False):
        assert reset is False
        calls.append(env.simulated_time)
        if env.simulated_time > 0.05:
            raise error
        return {
            "schema_version": 1,
            "profile": "native_coulomb_v1",
            "epoch": 1,
            "substeps": 0,
            "totals": {},
        }

    env.take_contact_accounting = accounting
    if external_timeout:
        with pytest.raises(TimeoutError) as caught:
            execute_action(
                env, selected(env), clearance_config, trace_callback=lambda e, t: recorded.append(t)
            )
        assert caught.value is error
    else:
        result = execute_action(
            env, selected(env), clearance_config, trace_callback=lambda e, t: recorded.append(t)
        )
        assert not result.execution_valid
        assert result.reason == "contact_accounting_capture_failed_RuntimeError: nonfinite counters"
        assert result.after is not None
        assert len(result.evidence["phase_traces"]) == len(recorded)
    assert len(calls) == len(recorded) + 1  # Failing read never becomes a captured trace.
    assert env.release_calls == 1 and not env.attached_ids and env.reset_calls == 0


@pytest.mark.parametrize("payload", [None, {"totals": {"impulse": float("nan")}}])
def test_malformed_contact_accounting_does_not_silently_omit_evidence(clearance_config, payload):
    env = HangingClothFake()
    env.take_contact_accounting = lambda *, reset=False: payload
    result = execute_action(env, selected(env), clearance_config)
    assert not result.execution_valid
    assert result.reason.startswith("contact_accounting_capture_failed_ValueError:")
    assert result.evidence["phase_traces"] == [] and env.step_calls == 0
    assert env.release_calls == 1 and env.reset_calls == 0
