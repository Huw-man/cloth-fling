"""FlingBot-style transformed RGB/action maps; no Torch or simulator imports.

The prescribed motion is reference laydown. Predicted values represent immediate
coverage change, never a trajectory or a privileged-geometry observation.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from cloth_fling.contracts import GraspAction
from cloth_fling.selection import conservative_foreground_mask, pixels_to_world

ANGLES_DEG = np.linspace(-90, 90, 12)
SCALES = np.linspace(1.0, 2.75, 8)
CANONICAL_WIDTH_PX = 32.0


class ValuePredictor(Protocol):
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Input uint8 [96,64,64,3]; output finite [96,64,64] immediate values."""
        ...


@dataclass(frozen=True)
class SpatialGraspAction(GraspAction):
    encoding: dict = field(default_factory=dict)


@dataclass
class SpatialActionBatch:
    rgb: np.ndarray
    feasible: np.ndarray
    to_original: np.ndarray
    observation_id: str
    crop: dict

    def decode(self, transform, u, v):
        if not (0 <= transform < len(self.to_original) and 0 <= u < 64 and 0 <= v < 64):
            raise ValueError("action-map index outside bounds")
        pair = np.array([[u - CANONICAL_WIDTH_PX / 2, v, 1], [u + CANONICAL_WIDTH_PX / 2, v, 1]])
        return (pair @ self.to_original[transform].T)[:, :2]

    def encode(self, pixels, transform):
        pair = np.asarray(pixels, dtype=float)
        if pair.shape != (2, 2) or not np.isfinite(pair).all():
            raise ValueError("invalid pixel pair")
        canonical = np.c_[pair, np.ones(2)] @ np.linalg.inv(self.to_original[transform]).T
        if not np.allclose(canonical[1, :2] - canonical[0, :2], [CANONICAL_WIDTH_PX, 0], atol=1e-7):
            raise ValueError("pair not represented by this transform")
        return canonical[:, :2].mean(axis=0)

    def action(self, index, policy):
        transform, v, u = map(int, index)
        if not self.feasible[transform, v, u]:
            raise ValueError("infeasible spatial action")
        pixels = self.decode(transform, u, v)
        angle = float(ANGLES_DEG[transform // len(SCALES)])
        scale = float(SCALES[transform % len(SCALES)])
        return SpatialGraspAction(
            pixels=tuple(map(tuple, pixels)),
            primitive="laydown",
            policy=policy,
            encoding={
                "codec": "reference_spatial_v1",
                "observation_id": self.observation_id,
                "map_index_tvu": [transform, v, u],
                "angle_deg": angle,
                "scale": scale,
                "center_uv": pixels.mean(axis=0).tolist(),
                "width_px": float(np.linalg.norm(pixels[1] - pixels[0])),
                "canonical_width_px": CANONICAL_WIDTH_PX,
                "crop": self.crop,
                "to_original": self.to_original[transform].tolist(),
            },
        )


def build_action_batch(observation, config):
    foreground = conservative_foreground_mask(observation)
    vu = np.argwhere(foreground)
    if not len(vu):
        raise ValueError("no RGB cloth foreground")
    low, high = vu[:, ::-1].min(axis=0), vu[:, ::-1].max(axis=0)
    center = (low + high) / 2
    extent = float(max(high - low + 1) * 1.5)
    yy, xx = np.mgrid[:64, :64]
    grid = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    images, masks, matrices = [], [], []
    h, w = foreground.shape

    def inside(uv):
        return (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)

    for angle in ANGLES_DEG:
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for scale in SCALES:
            matrix = np.eye(3)
            matrix[:2, :2] = rotation * extent / (64 * scale)
            matrix[:2, 2] = center - matrix[:2, :2] @ np.array([31.5, 31.5])
            original = (grid @ matrix.T)[..., :2]
            ij = np.rint(original).astype(int)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            good = inside(original)
            image[good] = observation.rgb[ij[..., 1][good], ij[..., 0][good]]
            shift = matrix[:2, 0] * CANONICAL_WIDTH_PX / 2
            left, right = original - shift, original + shift
            valid = inside(left) & inside(right)
            for endpoint in (left, right):
                uv = np.rint(endpoint).astype(int)
                safe = np.clip(uv, [0, 0], [w - 1, h - 1])
                valid &= foreground[safe[..., 1], safe[..., 0]]
            ids = np.flatnonzero(valid)
            if len(ids):
                a, b = left.reshape(-1, 2)[ids], right.reshape(-1, 2)[ids]
                # Privileged depth only builds feasibility; it never enters RGB tensors.
                depth = np.asarray(observation.depth)
                if depth.shape != (h, w):
                    raise ValueError("execution feasibility requires matching depth")
                ia, ib = np.rint(a).astype(int), np.rint(b).astype(int)
                z1, z2 = depth[ia[:, 1], ia[:, 0]], depth[ib[:, 1], ib[:, 0]]
                depth_ok = np.isfinite(z1) & np.isfinite(z2) & (z1 > 0) & (z2 > 0)
                valid.flat[ids[~depth_ok]] = False
                ids = ids[depth_ok]
                if len(ids):
                    xyz1 = pixels_to_world(observation, left.reshape(-1, 2)[ids])
                    xyz2 = pixels_to_world(observation, right.reshape(-1, 2)[ids])
                    separation = np.linalg.norm(xyz2 - xyz1, axis=1)
                    snap = 1.5 * config.mesh_spacing_m
                    feasible = (separation >= config.min_grasp_separation_m + 2 * snap) & (
                        separation <= config.max_grasp_separation_m - 2 * snap
                    )
                    xmin, xmax, ymin, ymax = config.table_bounds_m
                    for points in (xyz1, xyz2):
                        feasible &= (
                            (points[:, 0] >= xmin + snap)
                            & (points[:, 0] <= xmax - snap)
                            & (points[:, 1] >= ymin + snap)
                            & (points[:, 1] <= ymax - snap)
                        )
                    valid.flat[ids[~feasible]] = False
            images.append(image)
            masks.append(valid)
            matrices.append(matrix)
    return SpatialActionBatch(
        np.stack(images),
        np.stack(masks),
        np.stack(matrices),
        observation.observation_id,
        {
            "center_uv": center.tolist(),
            "extent_px": extent,
            "sampling": "nearest_pixel_center",
            "image_size": 64,
        },
    )


def choose_action(batch, *, rng=None, predictor=None):
    if not batch.feasible.any():
        return None
    if predictor is None:
        if rng is None:
            raise ValueError("uniform selection requires an explicit RNG")
        indices = np.flatnonzero(batch.feasible)
        flat = int(indices[rng.integers(len(indices))])
        policy = "uniform_valid_spatial"
    else:
        values = np.asarray(predictor(batch.rgb.copy()), dtype=float)
        if values.shape != batch.feasible.shape or not np.isfinite(values).all():
            raise ValueError("predictor must return finite [96,64,64] values")
        flat = int(np.argmax(np.where(batch.feasible, values, -np.inf)))
        policy = "learned_spatial"
    return batch.action(np.unravel_index(flat, batch.feasible.shape), policy)


def select_spatial_action(observation, config, *, rng, **_):
    return choose_action(build_action_batch(observation, config), rng=rng)


def learning_samples(dataset_dir):
    """Verified, settled-only RGB/mask/action/reward samples; no privileged fields leak.

    All attempts remain in the underlying dataset. This view deliberately excludes
    invalid/unsettled outcomes rather than turning failures into zero-reward labels.
    """
    from pathlib import Path

    from cloth_fling.data import verify_dataset
    from cloth_fling.data.reader import read_rgb, read_support, read_transitions

    root = Path(dataset_dir)
    verification = verify_dataset(root)
    if not verification["valid"]:
        raise ValueError("dataset verification failed")
    for row in read_transitions(root).to_pylist():
        if not row["execution_valid"] or row["reward"] is None:
            continue
        support = read_support(root, row["evidence_path"])
        if support.get("diagnostics", {}).get("settled") is not True:
            continue
        selection = support.get("evidence", {}).get("selection", {})
        if not selection or not selection.get("encoding"):
            continue
        yield {
            "rgb": read_rgb(root, row["before_rgb_path"]),
            "valid_action_mask": selection["valid_action_mask"].copy(),
            "action": selection["encoding"],
            "reward": float(row["reward"]),
            "split": row["split"],
            "reset_group_id": row["reset_group_id"],
        }
