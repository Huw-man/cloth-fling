"""Measured cloth evidence is distinct from commanded grasp motion (CPU only)."""

from dataclasses import replace

import numpy as np
import pytest
import test_primitives_clearance as clearance
from test_primitives_clearance import (
    HangingClothFake,
    phase_observer,
    selected,
)

from cloth_fling.sim.primitives import _phase_contact_deltas, execute_action


@pytest.fixture
def clearance_config():
    return clearance.clearance_config.__wrapped__()


def test_phase_contact_deltas_preserve_signed_work_and_large_counters(clearance_config):
    env = HangingClothFake()

    def accounting(*, reset=False):
        assert not reset
        return {
            "schema_version": 1,
            "profile": "native_coulomb_v2",
            "engine_version": "1.4.4",
            "epoch": 17,
            "substeps": 2**40 + env.step_calls,
            "totals": {
                "normal_impulse_kg_m_s": 2**40 + env.step_calls * 0.125,
                "tangential_impulse_kg_m_s": env.step_calls * 0.0625,
                "friction_work_j": -env.step_calls * 0.25,
                "free_friction_events": 2**40 + env.step_calls,
            },
        }

    env.take_contact_accounting = accounting
    result = execute_action(env, selected(env, "fling"), clearance_config)
    assert result.execution_valid, result.reason
    intervals = result.diagnostics["phase_contact"]
    assert sum(d["substeps"] for d in intervals) == env.step_calls
    for d in intervals:
        assert d["available"]
        assert d["totals_delta"]["normal_impulse_kg_m_s"] == d["substeps"] * 0.125
        assert d["totals_delta"]["friction_work_j"] == -d["substeps"] * 0.25
        assert d["totals_delta"]["free_friction_events"] == d["substeps"]
    for phase in ("clearance", "swing", "pullback", "lower", "place", "release", "settle"):
        assert any(d["phase"] == phase for d in intervals)
    assert intervals[-1]["ends_at_next_phase_boundary"] is False
    assert intervals[-1]["end_simulated_seconds"] == pytest.approx(result.simulated_seconds)
    assert result.diagnostics["settled_coverage"] == result.coverage_after


@pytest.mark.parametrize(
    "changed", ["epoch", "profile", "schema_version", "engine_version", "substeps", "totals"]
)
def test_phase_contact_epoch_identity_and_counter_discontinuities_are_unavailable(changed):
    a = {
        "schema_version": 1,
        "profile": "native_coulomb_v2",
        "engine_version": "1.4.4",
        "epoch": 3,
        "substeps": 100,
        "totals": {"friction_work_j": -4.0},
    }
    b = {
        **a,
        changed: {
            "epoch": 4,
            "profile": "legacy",
            "schema_version": 2,
            "engine_version": "0.4.3",
            "substeps": 0,
            "totals": {"different": 1},
        }[changed],
    }
    traces = [
        {"phase": "swing", "simulated_seconds": 0.0, "table_contact_accounting": a},
        {"phase": "pullback", "simulated_seconds": 1.0, "table_contact_accounting": b},
    ]
    result = _phase_contact_deltas(traces)[0]
    assert not result["available"] and result["totals_delta"] is None


def test_cloth_motion_and_tracking_error_measured_between_decimated_traces(clearance_config):
    env = HangingClothFake()
    disturbed = False

    def perturb(backend, free):
        nonlocal disturbed
        # One free-cloth impulse during the swing, absent from boundary samples.
        if backend.phase_hint == "swing" and not disturbed:
            backend.positions[np.flatnonzero(free)[0], 0] += 0.003
            disturbed = True
        # Deterministic backend tracking error; primitive never corrects free cloth.
        backend.positions[list(backend.attached_ids), 0] += 0.001

    env.disturb = perturb
    config = replace(clearance_config, trace_interval_seconds=30)
    result = execute_action(
        env,
        selected(env, "fling"),
        config,
        trace_callback=phase_observer,
    )
    assert result.execution_valid, result.reason
    traces = result.evidence["phase_traces"]
    phase = result.diagnostics["phase_motion"]["swing"]
    assert disturbed
    assert phase["max_free_cloth_rms_speed_m_s"] > max(
        t["free_cloth_rms_speed_m_s"] for t in traces if t["phase"] == "swing"
    )
    assert phase["max_attachment_error_m"] == pytest.approx(0.001)
    assert phase["min_free_cloth_airborne_fraction_proxy"] == 1
    assert phase["cloth_mean_particle_path_m"] > 0
    assert result.evidence["grasp_particle_ids"]
    for t in traces:
        positions = t["cloth_positions_m"]
        np.testing.assert_allclose(t["cloth_centroid_m"], positions.mean(axis=0))
        if t["attachment_ids"]:
            assert t["free_particle_count"] == len(positions) - 2
            hand = np.asarray(t["commanded_grasp_positions_m"])
            # Whole-cloth motion evidence includes the hanging free particles.
            if t["phase"] in {"clearance", "swing", "pullback"}:
                assert t["cloth_centroid_m"][2] < hand[:, 2].mean()
        if t["phase"] in {"released", "settle"}:
            assert t["attachment_error_max_m"] is None
            assert t["grasp_particle_ids"] == result.evidence["grasp_particle_ids"]
    assert env.reset_calls == 0 and env.release_calls == 1
