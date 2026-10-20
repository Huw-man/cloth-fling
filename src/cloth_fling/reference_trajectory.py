"""Versioned leading-edge placement. Polynomial bounds are computed before execution."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.polynomial import polynomial as poly


@dataclass(frozen=True)
class LeadingTrajectory:
    version: str = "reference_leading_v2"
    transfer_x_m: float = 0.45
    descent_seconds: float = 2.0
    transfer_seconds: float = 1.5
    drag_seconds: float = 0.8
    endpoint_hold_seconds: float = 1.0
    join_speed_m_s: float = 0.1
    max_speed_m_s: float = 0.8
    max_acceleration_m_s2: float = 2.0

    def __post_init__(self):
        if not all(np.isfinite(v) for k, v in asdict(self).items() if k != "version"):
            raise ValueError("nonfinite trajectory")
        if (
            self.version != "reference_leading_v2"
            or self.transfer_x_m not in (0.45, 0.55, 0.35, 0.65)
            or self.descent_seconds not in (2, 3, 4, 1.5, 1)
            or self.transfer_seconds != 1.5
            or self.drag_seconds != 0.8
            or self.endpoint_hold_seconds != 1
            or self.join_speed_m_s != 0.1
            or self.max_speed_m_s != 0.8
            or self.max_acceleration_m_s2 != 2
        ):
            raise ValueError("trajectory differs from authorized grid/bounds")


@dataclass
class Quintic:
    name: str
    seconds: float
    coefficients: np.ndarray  # ascending powers in normalized time, XYZ

    @classmethod
    def between(cls, name, seconds, start, end, start_velocity, end_velocity):
        a = np.zeros((6, 3))
        a[0], a[1] = start, np.asarray(start_velocity) * seconds
        a[3:] = np.linalg.solve(
            [[1, 1, 1], [3, 4, 5], [6, 12, 20]],
            [np.asarray(end) - a[0] - a[1], np.asarray(end_velocity) * seconds - a[1], [0, 0, 0]],
        )
        return cls(name, seconds, a)

    def at(self, seconds, derivative=0):
        a = poly.polyder(self.coefficients, m=derivative, axis=0) / self.seconds**derivative
        return poly.polyval(np.asarray(seconds) / self.seconds, a)

    def maximum_norm(self, derivative):
        a = poly.polyder(self.coefficients, m=derivative, axis=0) / self.seconds**derivative
        squared = np.zeros(1)
        for c in a.T:
            squared = poly.polyadd(squared, poly.polymul(c, c))
        roots = poly.polyroots(poly.polyder(squared))
        candidates = [0.0, 1.0] + [r.real for r in roots if abs(r.imag) < 1e-7 and 0 < r.real < 1]
        return float(np.linalg.norm(poly.polyval(candidates, a), axis=0).max())


def placement_schedule(config, dt=0.02):
    """Scale the path analytically, preserving C2 joins on the controller clock."""
    points = np.array(
        [[0.10, 0, 0.95], [config.transfer_x_m, 0, 0.95], [0.75, 0, 0.31], [0.80, 0, 0.31]]
    )
    scale = 1.0
    for _ in range(100):
        durations = (
            np.ceil(
                np.array([config.transfer_seconds, config.descent_seconds, config.drag_seconds])
                * scale
                / dt
                - 1e-10
            )
            * dt
        )
        velocities = np.array(
            [
                [0, 0, 0],
                [config.join_speed_m_s / scale, 0, 0],
                [config.join_speed_m_s / scale, 0, 0],
                [0, 0, 0],
            ]
        )
        segments = [
            Quintic.between(
                name, seconds, points[i], points[i + 1], velocities[i], velocities[i + 1]
            )
            for i, (name, seconds) in enumerate(zip(("transfer", "descent", "drag"), durations))
        ]
        speed = max(s.maximum_norm(1) for s in segments)
        acceleration = max(s.maximum_norm(2) for s in segments)
        factor = max(
            speed / config.max_speed_m_s, np.sqrt(acceleration / config.max_acceleration_m_s2)
        )
        if factor <= 1 + 1e-10:
            # Reserve attach, lift, hold, endpoint hold, release, full unsupported settle.
            fixed = 0.2 + 2 + 3 + sum(durations) + config.endpoint_hold_seconds + 0.2 + 5
            allowance = min(20.0, np.floor((35 - fixed + 1e-9) / dt) * dt)
            if allowance < 0:
                raise ValueError("complete phase schedule exceeds action budget")
            return segments, {
                "segments": [
                    {"phase": s.name, "seconds": s.seconds, "coefficients": s.coefficients.tolist()}
                    for s in segments
                ],
                "peak_speed_m_s": speed,
                "peak_acceleration_m_s2": acceleration,
                "tauten_allowance_seconds": allowance,
                "reserved_after_tauten_seconds": fixed - 2.2,
                "maximum_action_seconds": fixed + allowance,
            }
        scale *= factor * (1 + 1e-6)
    raise ValueError("analytic time scaling did not converge")


def candidate_grid():
    return [
        LeadingTrajectory(transfer_x_m=x, descent_seconds=t)
        for x in (0.45, 0.55, 0.35, 0.65)
        for t in (2, 3, 4, 1.5, 1)
    ]


def leading_fraction(positions, free, midpoint_x):
    p = np.asarray(positions)[free]
    if not len(p):
        raise ValueError("leading ordering requires free vertices")
    return float(np.mean(p[:, 0] <= midpoint_x + 0.02))
