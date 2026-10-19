"""The table adapter must be dissipative, local to support, and independent of actions."""

import numpy as np

from cloth_fling.sim.contact import table_friction_velocities


def apply(p, v, free, **kwargs):
    return table_friction_velocities(
        p,
        v,
        free=free,
        table_size=1.2,
        radius=0.005,
        dt=kwargs.get("dt", 0.002),
        static_friction=kwargs.get("mu", 0.15),
        kinetic_friction=kwargs.get("mu", 0.15),
    )


def test_coulomb_impulse_does_not_change_airborne_attached_outside_or_normal_state():
    p = np.array([[0, 0, 0.005], [0, 0, 0.02], [0, 0, 0.005], [0.7, 0, 0.005]])
    v = np.tile([0.1, 0.2, -0.03], (4, 1))
    result, changed = apply(p, v, [True, True, False, True])
    np.testing.assert_array_equal(changed, [0])
    np.testing.assert_array_equal(result[1:], v[1:])
    np.testing.assert_array_equal(result[:, 2], v[:, 2])
    assert np.linalg.norm(result[0, :2]) < np.linalg.norm(v[0, :2])


def test_friction_never_injects_kinetic_energy_and_zero_friction_is_identity():
    rng = np.random.default_rng(1)
    p = np.zeros((100, 3))
    p[:, 2] = 0.005
    v = rng.normal(size=(100, 3))
    result, _ = apply(p, v, np.ones(100, dtype=bool))
    assert np.all(np.sum(result**2, axis=1) <= np.sum(v**2, axis=1))
    zero, changed = apply(p, v, np.ones(100, dtype=bool), mu=0)
    np.testing.assert_array_equal(zero, v)
    assert len(changed) == 0


def test_contact_deceleration_uses_physical_dt_and_stops_without_reversing():
    p = np.array([[0, 0, 0.005]])
    v = np.array([[0.1, 0, 0]])
    full, _ = apply(p, v, [True], dt=0.004)
    half, _ = apply(p, v, [True])
    half, _ = apply(p, half, [True])
    np.testing.assert_allclose(full, half)
    np.testing.assert_allclose(0.1 - full[0, 0], 0.15 * 9.81 * 0.004)
    stopped, _ = apply(p, np.array([[0.0001, 0, 0]]), [True])
    assert stopped[0, 0] == 0
