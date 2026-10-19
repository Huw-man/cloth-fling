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
    p, v = np.asarray(positions), np.asarray(velocities)
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
