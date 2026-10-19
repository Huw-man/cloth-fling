import importlib
import subprocess
import sys

import numpy as np
import pytest

from cloth_fling.sim.mesh import (
    mesh_edges,
    mesh_identity,
    pixel_ray,
    ray_surface_hit,
    resolve_grasps,
    square_mesh,
    visible_particle,
)


def calibration(size=101):
    return {"intrinsics": [[100, 0, 50.5], [0, 100, 50.5], [0, 0, 1]],
            "camera_to_world": [[1, 0, 0, 0], [0, -1, 0, 0],
                                [0, 0, -1, 1], [0, 0, 0, 1]],
            "image_size": [size, size], "pixel_center_offset": 0.5}


def test_import_does_not_load_simulator():
    code = ("import sys; import cloth_fling.sim.backend; "
            "assert 'genesis' not in sys.modules; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_square_topology_and_hash():
    p, f = square_mesh()
    assert p.shape == (2601, 3)
    assert f.shape == (5000, 3)
    cross = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    assert np.all(cross[:, 2] > 0)
    assert cross[:, 2].sum() / 2 == pytest.approx(0.25)
    assert np.unique(f).size == len(p)
    edges = mesh_edges(f)
    assert len(edges) == 7600
    assert np.all(edges[:, 0] < edges[:, 1])
    assert mesh_identity(p, f) == mesh_identity(p.copy(), f.copy())
    moved = p.copy()
    moved[0, 0] += 0.001
    assert mesh_identity(p, f) != mesh_identity(moved, f)


@pytest.mark.parametrize("size,spacing", [(0, .01), (.5, -1), (np.nan, .1)])
def test_invalid_mesh(size, spacing):
    with pytest.raises(ValueError):
        square_mesh(size, spacing)


def test_camera_pixel_centres_and_axes():
    origin, ray = pixel_ray((50, 50), calibration())
    np.testing.assert_equal(origin, [0, 0, 1])
    np.testing.assert_allclose(ray, [0, 0, -1])
    assert pixel_ray((60, 60), calibration())[1][0] > 0
    assert pixel_ray((60, 60), calibration())[1][1] < 0
    for pixel in [(-1, 0), (0, 101), (np.nan, 0)]:
        with pytest.raises(ValueError):
            pixel_ray(pixel, calibration())


def test_nearest_surface_and_hidden_layer():
    # Hidden layer has a vertex exactly under the ray. The upper triangle's
    # vertices are further away but MUST be chosen from that visible layer.
    top = np.array([[-.1, -.1, .2], [.1, -.1, .2], [0, .1, .2]])
    hidden = np.array([[0, 0, .19], [.2, -.2, .19], [-.2, -.2, .19]])
    p = np.concatenate((hidden, top))
    f = np.array([[0, 1, 2], [3, 4, 5]])
    hit, face = ray_surface_hit(np.array([0, 0, 1]), np.array([0, 0, -1]), p, f)
    assert face == 1
    np.testing.assert_allclose(hit, [0, 0, .2])
    assert visible_particle((50, 50), p, f, calibration(), .15) == 5
    with pytest.raises(ValueError, match="radius"):
        visible_particle((50, 50), p, f, calibration(), .02)
    with pytest.raises(ValueError, match="hit"):
        visible_particle((100, 0), p, f, calibration(), .15)


def test_visible_distance_tie_breaks_by_id():
    p = np.array([[-.1, -.1, 0], [.1, -.1, 0], [.1, .1, 0], [-.1, .1, 0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    assert visible_particle((50, 50), p, f, calibration(), .2) == 0


def test_pair_validation():
    p, f = square_mesh(.5, .05)
    options = {"radius": .08, "workspace": .9, "min_separation": .08, "max_separation": .55}
    pair = ((30, 50), (70, 50))
    ids = resolve_grasps(pair, p, f, calibration(), **options)
    assert ids[0] != ids[1]
    with pytest.raises(ValueError, match="coincident"):
        resolve_grasps(((50, 50), (50, 50)), p, f, calibration(), **options)
    with pytest.raises(ValueError, match="workspace"):
        resolve_grasps(pair, p, f, calibration(), **(options | {"workspace": .1}))
    with pytest.raises(ValueError, match="separation"):
        resolve_grasps(pair, p, f, calibration(), **(options | {"max_separation": .1}))
    with pytest.raises(ValueError):
        resolve_grasps(((50, 50),), p, f, calibration(), **options)


def test_config_validation_before_genesis_import():
    module = importlib.import_module("cloth_fling.sim.backend")
    from cloth_fling.contracts import SceneConfig
    with pytest.raises(ValueError):
        module.GenesisBackend(SceneConfig(sim_dt=-1))
