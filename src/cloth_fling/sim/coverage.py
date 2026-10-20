"""Projected triangle union on a fixed, world-aligned metric sample grid."""

from __future__ import annotations

import numpy as np


class CoverageGrid:
    """Cell-centre rasterization, union by boolean OR, including overlap once.

    The finite grid covers the table. Triangles outside it are clipped; area
    resolution is cell_size**2. Both cloth and flat reference use this same grid.
    """

    def __init__(self, table_size: float = 1.2, cell_size: float = 0.0025, *, bounds=None):
        if not np.isfinite([table_size, cell_size]).all() or min(table_size, cell_size) <= 0:
            raise ValueError("grid dimensions must be positive and finite")
        if bounds is None:
            bounds = (-table_size / 2, table_size / 2, -table_size / 2, table_size / 2)
        xmin, xmax, ymin, ymax = bounds
        if not np.isfinite(bounds).all() or xmin >= xmax or ymin >= ymax:
            raise ValueError("invalid coverage bounds")
        self.n = int(np.ceil((xmax - xmin) / cell_size))
        self.ny = int(np.ceil((ymax - ymin) / cell_size))
        if max(self.n, self.ny) > 4096:
            raise ValueError("coverage grid too large")
        self.cell_size = (xmax - xmin) / self.n
        self.cell_height = (ymax - ymin) / self.ny
        self.minimum = xmin
        self.centres = self.minimum + (np.arange(self.n) + 0.5) * self.cell_size

        self.y_centres = ymin + (np.arange(self.ny) + 0.5) * self.cell_height

    def mask(self, positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        faces = np.asarray(faces)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("positions must be finite (N,3)")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype.kind not in "iu"
            or np.any(faces < 0)
            or np.any(faces >= len(positions))
        ):
            raise ValueError("invalid triangle indices")
        result = np.zeros((self.ny, self.n), dtype=bool)
        for tri in positions[faces, :2]:
            a, b, c = tri
            e, f = b - a, c - a
            det = e[0] * f[1] - e[1] * f[0]
            if abs(det) < 1e-15:
                continue
            lo = [np.searchsorted(self.centres, tri[:, 0].min(), side="left"),
                  np.searchsorted(self.y_centres, tri[:, 1].min(), side="left")]
            hi = [np.searchsorted(self.centres, tri[:, 0].max(), side="right"),
                  np.searchsorted(self.y_centres, tri[:, 1].max(), side="right")]
            x = self.centres[lo[0] : hi[0]][None, :] - a[0]
            y = self.y_centres[lo[1] : hi[1]][:, None] - a[1]
            u = (x * f[1] - y * f[0]) / det
            v = (e[0] * y - e[1] * x) / det
            result[lo[1] : hi[1], lo[0] : hi[0]] |= (
                (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
            )
        return result

    def area(self, positions: np.ndarray, faces: np.ndarray) -> float:
        return float(self.mask(positions, faces).sum() * self.cell_size * self.cell_height)

    def measure(
        self, positions: np.ndarray, faces: np.ndarray, rest_positions: np.ndarray
    ) -> tuple[float, float, float]:
        flat = self.area(rest_positions, faces)
        if flat <= 0:
            raise ValueError("flat reference has zero projected area")
        area = self.area(positions, faces)
        return area / flat, area, flat
