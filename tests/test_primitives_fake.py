"""CPU cloth fake: gravity, distance constraints, table collision, two kinematic attachments.

This is lifecycle/measurement test infrastructure, not a simulator/behavior qualification.
"""

import copy

import numpy as np

from cloth_fling.contracts import Observation, SceneConfig
from cloth_fling.selection import pixels_to_world


class FakeClothBackend:
    def __init__(self, *, metadata=True):
        self.config = SceneConfig(
            sim_dt=0.01,
            image_size=64,
            min_grasp_separation_m=0.07,
            max_grasp_separation_m=0.3,
            grasp_workspace_m=0.9,
        )
        xy = np.linspace(-0.1, 0.1, 5)
        self.rest_positions = np.array([[x, y, 0.004] for y in xy for x in xy])
        self.faces = np.array(
            [(i, i + 1, i + 5) for i in range(20) if i % 5 != 4]
            + [(i + 1, i + 6, i + 5) for i in range(20) if i % 5 != 4]
        )
        self.edges = np.unique(
            np.sort(
                np.vstack(
                    (self.faces[:, :2], self.faces[:, 1:], self.faces[:, [0, 2]])
                ),
                axis=1,
            ),
            axis=0,
        )
        self.lengths = np.linalg.norm(
            self.rest_positions[self.edges[:, 1]]
            - self.rest_positions[self.edges[:, 0]],
            axis=1,
        )
        self.positions = self.rest_positions.copy()
        self.velocities = np.zeros_like(self.positions)
        self.simulated_time = 0.0
        self.attached_ids = ()
        self.targets = None
        self.release_calls = self.reset_calls = self.step_calls = self.attach_calls = 0
        self.commands = []
        self.release_records = []
        self.metadata = metadata
        self.calibration = {
            "intrinsics": [[100, 0, 32], [0, 100, 32], [0, 0, 1]],
            "camera_to_world": [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, 1],
            ],
            "pixel_center_offset": 0.5,
            "image_size": [64, 64],
        }

    def get_positions(self):
        return self.positions.copy()

    def get_velocities(self):
        return self.velocities.copy()

    def observe(self):
        if not np.isfinite(self.positions).all():
            raise ValueError("cannot render nonfinite state")
        # Rasterize triangle surfaces with a z buffer, matching primary perspective calibration.
        rgb = np.full((64, 64, 3), (64, 71, 77), dtype=np.uint8)
        depth = np.full((64, 64), 1.0)
        q = self.positions
        uv = np.column_stack(
            (
                100 * q[:, 0] / (1 - q[:, 2]) + 31.5,
                -100 * q[:, 1] / (1 - q[:, 2]) + 31.5,
            )
        )
        for face in self.faces:
            vertices = uv[face]
            lo = np.maximum(np.floor(vertices.min(axis=0)).astype(int), 0)
            hi = np.minimum(np.ceil(vertices.max(axis=0)).astype(int), 63)
            if (lo > hi).any():
                continue
            x, y = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            pixels = np.column_stack((x.ravel(), y.ravel()))
            matrix = np.vstack((vertices.T, np.ones(3)))
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            weights = np.linalg.solve(
                matrix, np.column_stack((pixels, np.ones(len(pixels)))).T
            )
            keep = (weights >= -1e-9).all(axis=0)
            pixels, weights = pixels[keep], weights[:, keep]
            z = weights.T @ (1 - q[face, 2])
            u, v = pixels.T
            closer = z <= depth[v, u]
            u, v, z = u[closer], v[closer], z[closer]
            rgb[v, u], depth[v, u] = (200, 60, 40), z
        return Observation(
            f"fake-{self.simulated_time:.8f}",
            rgb,
            depth,
            copy.deepcopy(self.calibration),
        )

    def attach(self, pixels):
        self.attach_calls += 1
        points = pixels_to_world(self.observe(), np.asarray(pixels))
        ids = tuple(
            int(np.argmin(np.linalg.norm(self.positions - p, axis=1))) for p in points
        )
        self.attached_ids = ids
        self.targets = self.positions[list(ids)].copy()
        return ids

    def set_attachment_targets(self, targets):
        assert len(self.attached_ids) == 2
        self.targets = np.asarray(targets).copy()
        self.commands.append(
            (self.simulated_time, self.targets.copy(), self.attached_ids)
        )

    def step(self, count=1):
        for _ in range(count):
            self.step_calls += 1
            dt = self.config.sim_dt
            previous = self.positions.copy()
            self.velocities[:, 2] -= 9.81 * dt
            self.positions += self.velocities * dt
            inverse_mass = np.ones(len(self.positions))
            inverse_mass[list(self.attached_ids)] = 0
            a, b = self.edges.T
            for _ in range(25):
                delta = self.positions[b] - self.positions[a]
                lengths = np.linalg.norm(delta, axis=1)
                inv = inverse_mass[a] + inverse_mass[b]
                correction = (
                    delta
                    * ((lengths - self.lengths) / np.maximum(lengths * inv, 1e-12))[
                        :, None
                    ]
                ) * 0.5
                updates = np.zeros_like(self.positions)
                np.add.at(updates, a, correction * inverse_mass[a, None])
                np.add.at(updates, b, -correction * inverse_mass[b, None])
                self.positions += updates
                self.positions[:, 2] = np.maximum(self.positions[:, 2], 0.004)
                if self.attached_ids:
                    self.positions[list(self.attached_ids)] = self.targets
            self.velocities = (self.positions - previous) / dt
            contact = self.positions[:, 2] <= 0.004 + 1e-9
            self.velocities[contact, :2] *= (
                0.5  # fake table friction, not primitive damping
            )
            self.velocities[contact, 2] = np.maximum(self.velocities[contact, 2], 0)
            self.simulated_time += dt

    def release(self):
        self.release_calls += 1
        self.release_records.append(
            (
                self.simulated_time,
                self.attached_ids,
                self.positions.copy(),
                self.velocities.copy(),
            )
        )
        self.attached_ids = ()
        self.targets = None

    def coverage(self):
        # Convex projected footprint proxy sufficient for fake lifecycle tests.
        points = sorted(set(map(tuple, self.positions[:, :2])))

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower, upper = [], []
        for point in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        for point in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        hull = np.array(lower[:-1] + upper[:-1])
        area = (
            abs(
                np.sum(
                    hull[:, 0] * np.roll(hull[:, 1], -1)
                    - hull[:, 1] * np.roll(hull[:, 0], -1)
                )
            )
            / 2
        )
        return float(area / 0.04), float(area), 0.04

    def snapshot(self):
        return {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "simulated_time": self.simulated_time,
            "attached_ids": self.attached_ids,
            "targets": copy.deepcopy(self.targets),
            **(
                {"rest_positions": self.rest_positions, "edges": self.edges}
                if self.metadata
                else {}
            ),
        }

    def restore(self, state):
        self.positions = state["positions"].copy()
        self.velocities = state["velocities"].copy()
        self.simulated_time = state["simulated_time"]
        self.attached_ids = state["attached_ids"]
        self.targets = copy.deepcopy(state["targets"])
        return self.observe()

    def reset(self, *args, **kwargs):
        self.reset_calls += 1
        raise AssertionError("primitives/diagnostics must not reset")
