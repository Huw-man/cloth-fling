"""Coulomb friction for the finite horizontal table, absent from Genesis PBD/rigid coupling.

The adapter uses the resting normal load mg, not an impact-impulse estimate.
It is physics, never an action or an observation transformation.
"""

from __future__ import annotations

import numpy as np


def table_friction_velocities(
    positions,
    velocities,
    *,
    free,
    table_size,
    radius,
    dt,
    static_friction,
    kinetic_friction,
    gravity=9.81,
):
    """Apply a dissipative contact impulse; leave normal velocities and positions unchanged."""
    p, v = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
    free = np.asarray(free, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 3 or v.shape != p.shape or free.shape != (len(p),):
        raise ValueError("contact arrays must be N x 3 with an N-element free mask")
    values = [table_size, radius, dt, static_friction, kinetic_friction, gravity]
    if (
        not np.isfinite(p).all()
        or not np.isfinite(v).all()
        or not np.isfinite(values).all()
        or min(table_size, radius, dt, gravity) <= 0
        or min(static_friction, kinetic_friction) < 0
    ):
        raise ValueError("contact inputs must be finite with valid physical coefficients")
    support = (
        free
        & (p[:, 2] <= radius + 1e-6)
        & (p[:, 2] >= radius - 1e-4)
        & (v[:, 2] <= 1e-6)
        & (np.abs(p[:, :2]) <= table_size / 2).all(axis=1)
    )
    speed = np.linalg.norm(v[:, :2], axis=1)
    ids = np.flatnonzero(support & (speed > 0))
    result = v.copy()
    if len(ids):
        remaining = np.maximum(speed[ids] - kinetic_friction * gravity * dt, 0)
        remaining[speed[ids] <= static_friction * gravity * dt] = 0
        result[ids, :2] *= (remaining / speed[ids])[:, None]
    changed = np.flatnonzero(np.any(result != v, axis=1))
    return result, changed


def normal_impulse_friction(
    velocities, normals, normal_impulses, masses, *, free, static_friction, kinetic_friction
):
    """NumPy reference for native Coulomb friction after normal response.

    Unit normals and positive normal correction impulses are supplied by contact,
    never inferred from height, gravity or a velocity threshold. Zero impulse
    means no contact, including airborne particles. Returns velocity and impulse.
    This reference is offline validation, not a second runtime adapter.
    """
    v, n = np.asarray(velocities, dtype=float), np.asarray(normals, dtype=float)
    j, m = np.asarray(normal_impulses, dtype=float), np.asarray(masses, dtype=float)
    free = np.asarray(free, dtype=bool)
    if (
        v.ndim != 2
        or v.shape[1] != 3
        or n.shape != v.shape
        or j.shape != (len(v),)
        or m.shape != j.shape
        or free.shape != j.shape
    ):
        raise ValueError("invalid impulse array shapes")
    if (
        not all(np.isfinite(a).all() for a in (v, n, j, m))
        or np.any(m <= 0)
        or np.any(j < 0)
        or not np.allclose(np.linalg.norm(n, axis=1), 1)
        or not np.isfinite([static_friction, kinetic_friction]).all()
        or not 0 <= kinetic_friction <= static_friction
    ):
        raise ValueError("invalid impulse physics inputs")
    tangent = v - np.sum(v * n, axis=1)[:, None] * n
    speed = np.linalg.norm(tangent, axis=1)
    stop = m * speed
    amount = np.minimum(stop, kinetic_friction * j)
    amount[stop <= static_friction * j] = stop[stop <= static_friction * j]
    amount[~free] = 0
    impulse = (
        -tangent * np.divide(amount, speed, out=np.zeros_like(speed), where=speed > 0)[:, None]
    )
    return v + impulse / m[:, None], impulse
