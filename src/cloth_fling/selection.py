"""Visible RGB candidates with privileged depth used only for execution feasibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cloth_fling.contracts import GraspAction, Observation, SceneConfig


class NoValidPairError(ValueError):
    """No feasible visible pair was found within the declared search budget."""


@dataclass(frozen=True)
class SelectionConfig:
    min_grasp_separation_m: float = 0.08
    max_grasp_separation_m: float = 0.55
    grasp_workspace_m: float = 0.9
    mesh_spacing_m: float = 0.01
    foreground_threshold: float = 45.0  # RGB maximum channel, uint8 scale
    foreground_chroma: float = 20.0  # rejects the known nearly gray table
    min_pixel_separation: float = 2.0
    max_attempts: int = 4096
    max_boundary_candidates: int = 512

    @property
    def snap_uncertainty_m(self) -> float:
        """Per-endpoint surface-to-vertex snap bound agreed with the backend."""
        return 1.5 * self.mesh_spacing_m


def _options(config: SceneConfig | SelectionConfig) -> SelectionConfig:
    values = {k: getattr(config, k, v) for k, v in vars(SelectionConfig()).items()}
    c = SelectionConfig(**values)
    if not (0 < c.min_grasp_separation_m <= c.max_grasp_separation_m <= c.grasp_workspace_m):
        raise ValueError("invalid separation/workspace limits")
    if (
        not np.isfinite(list(values.values())).all()
        or c.max_attempts < 1
        or c.max_boundary_candidates < 2
        or c.mesh_spacing_m <= 0
        or c.min_pixel_separation <= 0
        or not isinstance(c.max_attempts, int)
        or not isinstance(c.max_boundary_candidates, int)
        or not 0 <= c.foreground_chroma <= 255
        or not 0 <= c.foreground_threshold <= 255
    ):
        raise ValueError("invalid selection configuration")
    return c


def foreground_mask(
    observation: Observation, threshold: float = 45, chroma: float = 20
) -> np.ndarray:
    """Known dark table assumption; no particle mask or learned segmentation."""
    rgb = np.asarray(observation.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("expected finite H x W x 3 RGB")
    scale = 255 if np.issubdtype(rgb.dtype, np.floating) and rgb.max(initial=0) <= 1 else 1
    high, low = np.max(rgb, axis=2) * scale, np.min(rgb, axis=2) * scale
    return (high > threshold) & (high - low > chroma)


def conservative_foreground_mask(
    observation: Observation, threshold: float = 45, chroma: float = 20
) -> np.ndarray:
    """One-pixel full 3x3 RGB erosion; image borders count as background.

    Rendering silhouettes can extend beyond the ray-tested surface. Require every
    neighbor, including diagonals, to be foreground before choosing a grasp pixel.
    """
    mask = foreground_mask(observation, threshold, chroma)
    height, width = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    safe = np.ones_like(mask)
    for dv in range(3):
        for du in range(3):
            safe &= padded[dv : dv + height, du : du + width]
    return safe


def pixels_to_world(observation: Observation, pixels: np.ndarray) -> np.ndarray:
    """Unproject visible pixels; see docs/primitives.md for calibration schemas."""
    uv = np.asarray(pixels, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError("pixels must be finite N x 2 (u,v)")
    h, w = observation.rgb.shape[:2]
    if np.any(uv < 0) or np.any(uv[:, 0] > w - 1) or np.any(uv[:, 1] > h - 1):
        raise ValueError("pixel outside image")
    if observation.depth is None or np.asarray(observation.depth).shape != (h, w):
        raise ValueError("privileged depth is required for execution feasibility")
    indices = np.rint(uv).astype(int)
    depth = np.asarray(observation.depth)[indices[:, 1], indices[:, 0]].astype(float)
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("depth must be positive finite optical depth in metres")
    c: dict[str, Any] = observation.calibration
    if "image_size" in c and list(c["image_size"]) != [w, h]:
        raise ValueError("calibration image_size differs from RGB")
    if "world_bounds_m" in c:
        xmin, xmax, ymin, ymax = np.asarray(c["world_bounds_m"], dtype=float)
        height = float(c["camera_height_m"])
        if w < 2 or h < 2 or not xmin < xmax or not ymin < ymax:
            raise ValueError("invalid orthographic calibration")
        xyz = np.column_stack(
            (
                xmin + uv[:, 0] / (w - 1) * (xmax - xmin),
                ymax - uv[:, 1] / (h - 1) * (ymax - ymin),
                height - depth,
            )
        )
    elif "intrinsics" in c and "camera_to_world" in c:
        k = np.asarray(c["intrinsics"], dtype=float)
        transform = np.asarray(c["camera_to_world"], dtype=float)
        if (
            k.shape != (3, 3)
            or transform.shape != (4, 4)
            or not np.isfinite(k).all()
            or not np.isfinite(transform).all()
        ):
            raise ValueError("invalid pinhole calibration shapes")
        rays = np.linalg.solve(
            k,
            np.column_stack((uv + float(c.get("pixel_center_offset", 0.5)), np.ones(len(uv)))).T,
        ).T
        if not np.isfinite(rays).all() or np.any(rays[:, 2] <= 0):
            raise ValueError("invalid optical rays")
        camera = rays * (depth / rays[:, 2])[:, None]
        xyz = (np.column_stack((camera, np.ones(len(uv)))) @ transform.T)[:, :3]
    else:
        raise ValueError("unsupported calibration: world_bounds_m or intrinsics required")
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite unprojected geometry")
    return xyz


def _candidates(observation, config, boundary_only=False):
    c = _options(config)
    mask = conservative_foreground_mask(observation, c.foreground_threshold, c.foreground_chroma)
    if boundary_only:
        padded = np.pad(mask, 1)
        interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        mask &= ~interior
    vu = np.argwhere(mask)
    if observation.depth is None:
        raise NoValidPairError("privileged depth required; no RGB-only feasibility fallback")
    depth = np.asarray(observation.depth)
    if depth.shape != mask.shape:
        raise ValueError("depth shape differs from RGB")
    vu = vu[np.isfinite(depth[vu[:, 0], vu[:, 1]]) & (depth[vu[:, 0], vu[:, 1]] > 0)]
    uv = vu[:, ::-1]
    xyz = pixels_to_world(observation, uv)
    workspace_limit = c.grasp_workspace_m / 2 - c.snap_uncertainty_m
    keep = (np.abs(xyz[:, :2]) <= workspace_limit).all(axis=1)
    uv, xyz = uv[keep], xyz[keep]
    # Stable image-order sampling bounds the geometry baseline, never accesses hidden particles.
    if boundary_only and len(uv) > c.max_boundary_candidates:
        subset = np.linspace(0, len(uv) - 1, c.max_boundary_candidates, dtype=int)
        uv, xyz = uv[subset], xyz[subset]
    if len(uv) < 2:
        raise NoValidPairError("fewer than two feasible foreground pixels")
    return c, uv, xyz


def _valid(c, uv_a, uv_b, xyz_a, xyz_b):
    distance = np.linalg.norm(xyz_b - xyz_a, axis=-1)
    return (
        (distance >= c.min_grasp_separation_m + 2 * c.snap_uncertainty_m)
        & (distance <= c.max_grasp_separation_m - 2 * c.snap_uncertainty_m)
        & (np.linalg.norm(uv_b - uv_a, axis=-1) >= c.min_pixel_separation)
    )


def feasible_candidates(
    observation: Observation,
    config: SceneConfig | SelectionConfig,
    *,
    primitive: str = "laydown",
    max_pairs: int = 256,
) -> list[GraspAction]:
    """Widest feasible RGB boundary pairs, bounded candidates, stable lexicographic ties.

    Geometry/depth are privileged; this is an execution baseline, not a learned RGB policy.
    """
    if primitive not in {"laydown", "fling"} or max_pairs < 1:
        raise ValueError("invalid primitive or max_pairs")
    c, uv, xyz = _candidates(observation, config, boundary_only=True)
    pairs = []
    for i in range(len(uv) - 1):
        js = np.flatnonzero(_valid(c, uv[i], uv[i + 1 :], xyz[i], xyz[i + 1 :])) + i + 1
        pairs.extend((float(np.linalg.norm(xyz[j] - xyz[i])), i, int(j)) for j in js)
    pairs.sort(key=lambda item: (-item[0], tuple(uv[item[1]]), tuple(uv[item[2]])))
    return [
        GraspAction(
            (tuple(map(float, uv[i])), tuple(map(float, uv[j]))),
            primitive,
            "geometry_guided_privileged_depth",
        )
        for _, i, j in pairs[:max_pairs]
    ]


def select_action(
    observation: Observation,
    config: SceneConfig | SelectionConfig,
    *,
    policy: str = "uniform_valid",
    primitive: str = "laydown",
    rng: np.random.Generator | None = None,
) -> GraspAction:
    if primitive not in {"laydown", "fling"}:
        raise ValueError(f"unknown primitive: {primitive}")
    if policy in {"geometry_guided", "geometry_guided_privileged_depth"}:
        pairs = feasible_candidates(observation, config, primitive=primitive, max_pairs=1)
        if not pairs:
            raise NoValidPairError("no feasible visible boundary pair")
        return pairs[0]
    if policy != "uniform_valid":
        raise ValueError(f"unknown policy: {policy}")
    c, uv, xyz = _candidates(observation, config)
    rng = np.random.default_rng(0) if rng is None else rng
    for _ in range(c.max_attempts):
        i, j = sorted(rng.choice(len(uv), 2, replace=False))
        if _valid(c, uv[i], uv[j], xyz[i], xyz[j]):
            return GraspAction(
                (tuple(map(float, uv[i])), tuple(map(float, uv[j]))),
                primitive,
                "uniform_valid",
            )
    raise NoValidPairError(f"no valid pair found in {c.max_attempts} uniform attempts")
