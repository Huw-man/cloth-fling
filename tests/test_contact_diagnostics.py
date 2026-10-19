"""Offline diagnostic gates must catch normal energy injection, not hide it."""

import numpy as np
import pytest

from cloth_fling.contact_diagnostics import summarize_contact_trace


def test_normal_energy_gain_is_visible_independent_of_tangential_dissipation():
    p = np.zeros((5, 2, 3))
    p[:, :, 2] = 0.005
    v = np.zeros_like(p)
    v[0, :, 0] = 10
    v[2, :, 2] = 1
    r = summarize_contact_trace(p, v, dt=0.002, radius=0.005, mu_kinetic=0.3)
    assert not r["normal_energy_passed"]
    assert r["normal_energy_gain_j_kg"] == pytest.approx(0.5)
    assert r["total_energy_max_j_kg"] == 50


def test_trace_reports_analytic_sliding_and_drop_rebound():
    dt, mu = 0.001, 0.3
    t = np.arange(300) * dt
    p = np.zeros((len(t), 2, 3))
    p[:, :, 2] = 0.005
    v = np.zeros_like(p)
    v[:, :, 0] = np.maximum(0.5 - mu * 9.81 * t, 0)[:, None]
    p[:, :, 0] = np.cumsum(v[:, :, 0], axis=0) * dt
    r = summarize_contact_trace(p, v, dt=dt, radius=0.005, mu_kinetic=mu)
    assert r["deceleration_relative_error"] < 1e-10
    assert r["normal_energy_passed"] and r["total_energy_passed"]
    assert r["stopping_distance_m"] == pytest.approx(0.5**2 / (2 * mu * 9.81), abs=0.001)
