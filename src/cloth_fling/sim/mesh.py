"""Deterministic cloth topology and visible-surface picking (NumPy only)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def square_mesh(size: float = 0.5, spacing: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Return a centred, +z-facing square, with spacing no larger than requested."""
    if not np.isfinite([size, spacing]).all() or min(size, spacing) <= 0:
        raise ValueError("mesh size and spacing must be finite and positive")
    n = int(np.ceil(size / spacing))
    if n > 1000:
        raise ValueError("mesh exceeds bounded resolution")
    x, y = np.meshgrid(np.linspace(-size / 2, size / 2, n + 1),
                       np.linspace(-size / 2, size / 2, n + 1))
    positions = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    a = (np.arange(n)[:, None] * (n + 1) + np.arange(n)).ravel()
    faces = np.concatenate((np.column_stack((a, a + 1, a + n + 2)),
                            np.column_stack((a, a + n + 2, a + n + 1))))
    return positions, faces.astype(np.int32)


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                            faces[:, [2, 0]])), axis=1), axis=0)


def mesh_identity(positions: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array, dtype in ((positions, "<f8"), (faces, "<i8")):
        value = np.ascontiguousarray(array, dtype=dtype)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_obj(path: str | Path, positions: np.ndarray, faces: np.ndarray) -> None:
    """Write an unprocessed mesh; Genesis may reorder vertices, checked at build."""
    with Path(path).open("w") as stream:
        stream.writelines(f"v {x:.12g} {y:.12g} {z:.12g}\n" for x, y, z in positions)
        stream.writelines(f"f {a} {b} {c}\n" for a, b, c in faces + 1)


def pixel_ray(pixel: tuple[float, float], calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(pixel, dtype=float)
    width, height = calibration["image_size"]
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("pixel must contain two finite coordinates")
    if not (0 <= uv[0] < width and 0 <= uv[1] < height):
        raise ValueError("pixel outside image")
    pose = np.asarray(calibration["camera_to_world"], dtype=float)
    k = np.asarray(calibration["intrinsics"], dtype=float)
    local = np.linalg.solve(k, [*(uv + calibration.get("pixel_center_offset", 0.5)), 1])
    direction = pose[:3, :3] @ local
    return pose[:3, 3].copy(), direction / np.linalg.norm(direction)


def ray_surface_hit(origin: np.ndarray, direction: np.ndarray, positions: np.ndarray,
                    faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Double-sided nearest ray/triangle hit; triangle ID breaks coincident ties."""
    triangles = positions[faces]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
    s = origin - triangles[:, 0]
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv * (q @ direction)
    t = inv * np.einsum("ij,ij->i", e2, q)
    valid = ((np.abs(det) > 1e-12) & (u >= -1e-8) & (v >= -1e-8)
             & (u + v <= 1 + 1e-8) & (t > 1e-8))
    if not valid.any():
        raise ValueError("pixel does not hit cloth")
    ids = np.flatnonzero(valid)
    face_id = int(ids[np.argmin(t[ids])])
    return origin + t[face_id] * direction, face_id


def visible_particle(pixel: tuple[float, float], positions: np.ndarray, faces: np.ndarray,
                     calibration: dict, radius: float) -> int:
    """Nearest visible vertex of the FIRST hit triangle, within a metric radius.

    Restricting eligibility to that face prevents a closer hidden-layer vertex from
    winning. Each candidate is also ray-tested from the camera to reject occluded
    vertices at folds. Distance ties are broken by particle ID.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("pick radius must be positive")
    origin, direction = pixel_ray(pixel, calibration)
    hit, face = ray_surface_hit(origin, direction, positions, faces)
    eligible = np.sort(faces[face])
    distances = np.linalg.norm(positions[eligible] - hit, axis=1)
    for index in np.lexsort((eligible, distances)):
        particle = int(eligible[index])
        if distances[index] > radius:
            break
        ray = positions[particle] - origin
        length = np.linalg.norm(ray)
        visible_hit, _ = ray_surface_hit(origin, ray / length, positions, faces)
        if np.linalg.norm(visible_hit - positions[particle]) <= 1e-5:
            return particle
    raise ValueError("no visible eligible particle within pick radius")


def resolve_grasps(pixels: tuple, positions: np.ndarray, faces: np.ndarray,
                   calibration: dict, *, radius: float, workspace: float,
                   min_separation: float, max_separation: float) -> tuple[int, int]:
    if np.asarray(pixels).shape != (2, 2):
        raise ValueError("exactly two image pixels required")
    ids = tuple(visible_particle(p, positions, faces, calibration, radius) for p in pixels)
    if ids[0] == ids[1]:
        raise ValueError("coincident grasp particle IDs")
    points = positions[list(ids)]
    if np.any(np.abs(points[:, :2]) > workspace / 2):
        raise ValueError("grasp outside workspace")
    separation = np.linalg.norm(points[0] - points[1])
    if not min_separation <= separation <= max_separation:
        raise ValueError("grasp separation outside bounds")
    return ids
