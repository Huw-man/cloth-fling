"""Reference rest geometry and material-line mapping, without simulator imports."""

import numpy as np


def triangle_area(positions, faces):
    triangles = positions[faces]
    return float(
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        ).sum()
        / 2
    )


def reference_mesh(config):
    """Frozen adaptation: two crossed 3-wave modes with seeded phases, max-normalized."""
    nx = int(round(config.cloth_size_m / config.mesh_spacing_m))
    ny = int(round(config.cloth_width_m / config.mesh_spacing_m))
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, nx + 1), np.linspace(-0.5, 0.5, ny + 1))
    phase = np.random.default_rng(config.wrinkle_seed).uniform(0, 2 * np.pi, 2)
    wave = np.sin(2 * np.pi * config.wrinkle_waves * (u + v) + phase[0]) + np.sin(
        2 * np.pi * config.wrinkle_waves * (u - v) + phase[1]
    )
    wave -= wave.mean()
    wave *= config.wrinkle_amplitude_m / max(float(np.abs(wave).max()), 1e-12)
    positions = np.c_[
        u.ravel() * config.cloth_size_m, v.ravel() * config.cloth_width_m, wave.ravel()
    ]
    a = (np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)).ravel()
    faces = np.concatenate((np.c_[a, a + 1, a + nx + 2], np.c_[a, a + nx + 2, a + nx + 1])).astype(
        np.int32
    )
    target = config.cloth_size_m * config.cloth_width_m
    low, high = 0.0, 1.0
    for _ in range(60):
        scale = (low + high) / 2
        trial = positions.copy()
        trial[:, :2] *= scale
        if triangle_area(trial, faces) > target:
            high = scale
        else:
            low = scale
    scale = (low + high) / 2
    positions[:, :2] *= scale
    return positions, faces, scale


def material_line(rest, faces, start, end, samples=65):
    """Map an XY material line onto native rest triangles using barycentric weights."""
    triangles = np.asarray(rest)[faces, :2]
    e, f = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    det = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
    valid = np.abs(det) > 1e-14
    ids, weights = [], []
    for point in np.linspace(start, end, samples):
        d = point[:2] - triangles[:, 0]
        b = np.divide(
            d[:, 0] * f[:, 1] - d[:, 1] * f[:, 0], det, out=np.full_like(det, -1e6), where=valid
        )
        c = np.divide(
            e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0], det, out=np.full_like(det, -1e6), where=valid
        )
        w = np.c_[1 - b - c, b, c]
        candidates = np.flatnonzero(valid & (w >= -1e-7).all(axis=1))
        if not len(candidates):
            raise ValueError("material line leaves native rest surface")
        face = int(candidates[0])
        ids.append(faces[face])
        weights.append(w[face])
    return np.asarray(ids), np.asarray(weights)


def sample_line(positions, indices, weights):
    return np.einsum("ij,ijk->ik", weights, positions[indices])
