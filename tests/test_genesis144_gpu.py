"""Opt-in static Genesis 1.4.4 checks; imports stay offline until explicitly run.

Requires the coordinator's exclusive GPU lease. No stepping, reset generation,
primitive, or collection is permitted. Historical replay additionally requires
explicit source paths; its new output always lives outside the source tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("CLOTH_FLING_RUN_GENESIS144_GPU") != "1",
        reason="set CLOTH_FLING_RUN_GENESIS144_GPU=1 only with the exclusive GPU lease",
    ),
]


def _forbidden(*args, **kwargs):
    pytest.fail("static Genesis 1.4.4 rendering checks must never step or reset physics")


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _report(directory, name, document):
    (directory / name).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


@pytest.fixture
def native_env(monkeypatch):
    # These imports execute only after the opt-in skip gate, never at collection.
    from cloth_fling.contracts import SceneConfig
    from cloth_fling.sim.appearance import AppearanceConfig
    from cloth_fling.sim.backend import GenesisBackend

    config = SceneConfig(
        contact_profile="native_coulomb_v2",
        cloth_size_m=0.2,
        mesh_spacing_m=0.02,
        image_size=256,
    )
    with GenesisBackend(config, appearance=AppearanceConfig()) as env:
        assert env._gs.__version__ == "1.4.4"
        assert env._gs.device.type == "cuda"
        monkeypatch.setattr(env.scene, "step", _forbidden)
        monkeypatch.setattr(env.scene, "reset", _forbidden)
        yield env


def _project(points, calibration):
    inverse = np.linalg.inv(np.asarray(calibration["camera_to_world"]))
    camera = np.c_[points, np.ones(len(points))] @ inverse.T
    uvz = camera[:, :3] @ np.asarray(calibration["intrinsics"]).T
    return tuple(map(tuple, uvz[:, :2] / uvz[:, 2:] - calibration["pixel_center_offset"]))


def _clean_native(env):
    assert env.camera.debug is False
    rgb = env.camera.render(rgb=True, depth=False, antialiasing=False, force_render=True)[0]
    depth = env.camera.render(rgb=False, depth=True, antialiasing=False, force_render=True)[1]
    return _array(rgb), _array(depth)


def _checked_capture(env, monkeypatch, *, phase, selected=None, fail=False):
    """Inspect native draws and prove cache/state/camera preservation even on error."""
    from cloth_fling.rendering import MARKER_COLORS, MARKER_RADIUS_M, render_debug_views

    baseline = env.observe()
    cached = env._observation
    cache_arrays = (cached.rgb, cached.depth)
    state_hash, step = env.snapshot()["state_hash"], env.scene._t
    pose = env.camera.transform.copy()
    fields = {
        k: _array(getattr(env.camera, k)) for k in ("_pos", "_lookat", "_up", "_transform", "_quat")
    }
    targets = None if env._targets is None else env._targets.copy()
    buffers_ready = env._render_buffers_ready
    external = dict(env.scene.visualizer.context.external_nodes)
    original_render, original_draw = env.camera.render, env.scene.draw_debug_sphere
    draws, calls = [], []

    def checked_render(**options):
        assert env.camera.debug is True
        assert options["rgb"] is True and options["depth"] is False
        calls.append(options)
        if fail and len(calls) == 2:
            raise RuntimeError("injected native debug render failure")
        return original_render(**options)

    def checked_draw(**options):
        draws.append({**options, "pos": _array(options["pos"])})
        assert options["radius"] == MARKER_RADIUS_M == 0.01
        assert options["color"] == MARKER_COLORS[len(draws) - 1]
        return original_draw(**options)

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(env.camera, "render", checked_render)
            scoped.setattr(env.scene, "draw_debug_sphere", checked_draw)
            views = render_debug_views(env, phase=phase, selected_targets_m=selected)
    finally:
        assert env.camera.debug is False
        np.testing.assert_array_equal(env.camera.transform, pose)
        for name, value in fields.items():
            np.testing.assert_array_equal(_array(getattr(env.camera, name)), value)
        assert env.scene.visualizer.context.external_nodes == external
        assert env._render_buffers_ready is buffers_ready
        assert env._observation is cached
        assert env._observation.rgb is cache_arrays[0] and env._observation.depth is cache_arrays[1]
        np.testing.assert_array_equal(cached.rgb, baseline.rgb)
        np.testing.assert_array_equal(cached.depth, baseline.depth)
        assert env.snapshot()["state_hash"] == state_hash
        assert env.scene._t == step
        if targets is None:
            assert env._targets is None
        else:
            np.testing.assert_array_equal(env._targets, targets)
        rgb, depth = _clean_native(env)
        np.testing.assert_array_equal(rgb, baseline.rgb)
        np.testing.assert_array_equal(depth, baseline.depth)
        assert env._observation is cached
    assert calls
    return views, draws


def _added_marker_pixels(rgb, clean):
    changed = np.any(rgb != clean, axis=2)
    channels = rgb.astype(np.int16)
    red = (
        (channels[..., 0] > 100)
        & (channels[..., 0] > 2 * channels[..., 1])
        & (channels[..., 0] > 2 * channels[..., 2])
    )
    blue = (
        (channels[..., 2] > 100)
        & (channels[..., 2] > 2 * channels[..., 0])
        & (channels[..., 2] > 2 * channels[..., 1])
    )
    return {
        "red": int(np.count_nonzero(changed & red)),
        "blue": int(np.count_nonzero(changed & blue)),
    }


def test_matte_markers_camera_and_clean_policy_isolation(native_env, tmp_path, monkeypatch):
    from PIL import Image

    env = native_env
    report = {
        "passed": False,
        "physics_stepped": False,
        "runtime": env.runtime_identity,
        "rendering": env.rendering_identity,
        "material": [],
        "markers": {},
    }
    table = env.scene.rigid_solver.entities[0]
    for geom in table.vgeoms:
        node = env.scene.visualizer.context.rigid_nodes[geom.uid]
        for primitive in node.mesh.primitives:
            material = primitive.material
            report["material"].append(
                {
                    "roughness": float(material.roughnessFactor),
                    "metallic": float(material.metallicFactor),
                }
            )
            _report(tmp_path, "matte-debug-report.json", report)
            assert material.roughnessFactor == 0.8 and material.metallicFactor == 0
            assert material.baseColorTexture is not None
            texture = np.asarray(material.baseColorTexture.source)
            assert texture.dtype == np.uint8 and texture.shape[:2] == (128, 128)
            np.testing.assert_array_equal(texture[..., 0], texture[..., 1])
            np.testing.assert_array_equal(texture[..., 1], texture[..., 2])
            assert np.ptp(texture[..., 0]) > 0
            mean_gray = float(texture[..., :3].mean() / 255)
            report["material"][-1]["mean_texture_gray"] = mean_gray
            assert 0.53 < mean_gray < 0.57
            assert primitive.texcoord_0 is not None and np.isfinite(primitive.texcoord_0).all()
    assert report["material"]
    _report(tmp_path, "matte-debug-report.json", report)
    clean, draws = _checked_capture(env, monkeypatch, phase="released")
    assert not draws
    for view, rgb in clean.items():
        Image.fromarray(rgb).save(tmp_path / f"clean-{view}.png")
    positions = env.get_positions()
    ids = [
        int(np.argmin(np.linalg.norm(positions[:, :2] - point, axis=1)))
        for point in ((-0.06, 0), (0.06, 0))
    ]
    selected = positions[ids].copy()
    selected_views, draws = _checked_capture(
        env, monkeypatch, phase="pre_attach", selected=selected
    )
    np.testing.assert_array_equal(np.array([d["pos"] for d in draws]), selected)
    attached = env.attach(_project(selected, env.calibration))
    env.cloth.set_particles_vel(np.tile([0.05, -0.02, 0.03], (2, 1)), list(attached))
    measured = env.get_positions()[list(attached)]
    env.set_attachment_targets(measured + [0, 0, 0.1])  # Deliberately different; never executed.
    attached_views, draws = _checked_capture(
        env, monkeypatch, phase="lift", selected=selected + [0, 0, 0.2]
    )
    np.testing.assert_array_equal(np.array([d["pos"] for d in draws]), measured)
    velocity = env.get_velocities()
    env.release()
    np.testing.assert_array_equal(env.get_velocities(), velocity)
    released, draws = _checked_capture(env, monkeypatch, phase="released", selected=selected)
    assert not draws
    for phase, views in (
        ("selected", selected_views),
        ("attached", attached_views),
        ("released", released),
    ):
        report["markers"][phase] = {}
        for view, rgb in views.items():
            Image.fromarray(rgb).save(tmp_path / f"{phase}-{view}.png")
            counts = _added_marker_pixels(rgb, clean[view])
            report["markers"][phase][view] = counts
        _report(tmp_path, "matte-debug-report.json", report)
    for phase, views in report["markers"].items():
        for counts in views.values():
            if phase == "released":
                assert all(n == 0 for n in counts.values())
            else:
                assert all(n > 0 for n in counts.values())
    with pytest.raises(RuntimeError, match="injected native debug render failure"):
        _checked_capture(env, monkeypatch, phase="pre_attach", selected=selected, fail=True)
    report["passed"] = True
    _report(tmp_path, "matte-debug-report.json", report)


def _ray_depth_report(env, observation):
    """Independent centre rays through physical faces, including visible backfaces."""
    from cloth_fling.sim.mesh import pixel_ray, ray_surface_hit

    positions = env.get_positions()
    triangles = positions[env.faces]
    centroids = triangles.mean(axis=1)[np.linspace(0, len(triangles) - 1, 128, dtype=int)]
    pixels = sorted(
        set(map(tuple, np.rint(_project(centroids, observation.calibration)).astype(int)))
    )
    inverse = np.linalg.inv(np.asarray(observation.calibration["camera_to_world"]))
    errors, backfaces = [], 0
    for u, v in pixels:
        if not (0 <= u < env.config.image_size and 0 <= v < env.config.image_size):
            continue
        origin, direction = pixel_ray((u, v), observation.calibration)
        try:
            point, face_id = ray_surface_hit(origin, direction, positions, env.faces)
        except ValueError:
            continue
        triangle = triangles[face_id]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        backfaces += int(np.dot(normal, direction) > 0)
        expected = (inverse @ [*point, 1])[2]
        errors.append(abs(float(observation.depth[v, u]) - expected))
    return {
        "rays": len(errors),
        "visible_backface_rays": backfaces,
        "max_depth_error_m": float(max(errors)) if errors else None,
    }


def test_schema2_folded_same_step_roundtrip(native_env, tmp_path):
    from cloth_fling.sim.backend import _state_hash
    from cloth_fling.sim.mesh import finite_radius_fold

    env = native_env
    original = env.snapshot()
    before = env.observe()
    assert original["schema_version"] == 2 and original["engine_version"] == "1.4.4"
    step = env.scene._t
    changed = copy.deepcopy(original)
    changed["positions"] = finite_radius_fold(
        original["positions"], 2 * env.config.mesh_spacing_m, env._initial_height + 0.08
    ).astype(np.float32)
    changed["velocities"][:] = [0.02, -0.01, 0.03]
    changed["state_hash"] = _state_hash(changed)
    raised = env.restore(changed)
    np.savez_compressed(
        tmp_path / "folded-render.npz", rgb=raised.rgb, depth=raised.depth, **changed
    )
    rays = _ray_depth_report(env, raised)
    _report(tmp_path, "folded-ray-report.json", rays)
    assert rays["rays"] >= 16 and rays["visible_backface_rays"] > 0
    assert rays["max_depth_error_m"] < 0.001
    assert np.any(raised.rgb != before.rgb) and np.any(raised.depth != before.depth)
    assert env.snapshot()["state_hash"] == changed["state_hash"]
    np.savez_compressed(tmp_path / "schema2-roundtrip.npz", **original)
    with np.load(tmp_path / "schema2-roundtrip.npz", allow_pickle=False) as archive:
        restored = env.restore(dict(archive))
    assert env.scene._t == step
    assert env.snapshot()["state_hash"] == original["state_hash"]
    np.testing.assert_array_equal(env.get_positions(), original["positions"])
    np.testing.assert_array_equal(env.get_velocities(), original["velocities"])
    np.testing.assert_array_equal(restored.rgb, before.rgb)
    np.testing.assert_array_equal(restored.depth, before.depth)
    assert restored.observation_id == before.observation_id
    _report(
        tmp_path,
        "schema2-report.json",
        {
            "passed": True,
            "physics_stepped": False,
            "runtime": env.runtime_identity,
            "state_hash": original["state_hash"],
            "folded_ray_depth": rays,
        },
    )


def test_old_physical_rejection_and_render_only_latch(native_env, tmp_path):
    from cloth_fling.sim.backend import _json, _state_hash

    env = native_env
    current = env.snapshot()
    archival = copy.deepcopy(current)
    for key in ("engine", "engine_version", "runtime_identity_json", "substeps"):
        archival.pop(key)
    archival["schema_version"] = 1
    config = json.loads(archival["config_json"])
    config["contact_profile"] = "native_coulomb_v1"
    archival["config_json"] = _json(config)
    archival["config_hash"] = hashlib.sha256(archival["config_json"].encode()).hexdigest()
    archival["velocities"][:] = 0.5
    archival["free"][0] = False  # These physical fields must not be imported.
    archival["state_hash"] = _state_hash(archival)
    with pytest.raises(ValueError, match="schema-1 Genesis 0.4.3"):
        env.restore(archival)
    assert env.snapshot()["state_hash"] == current["state_hash"]
    step = env.scene._t
    env.restore_recorded_geometry(archival)
    assert env.capabilities["render_only"] and not env.capabilities["physical_restore"]
    np.testing.assert_array_equal(env.get_positions(), archival["positions"])
    np.testing.assert_array_equal(env.get_velocities(), 0)
    assert env._free().all() and not env.attached_ids
    assert env.scene._t == step
    for operation in (
        lambda: env.step(),
        lambda: env.reset(),
        lambda: env.snapshot(),
        lambda: env.restore(current),
        lambda: env.attach(((0, 0), (1, 1))),
    ):
        with pytest.raises(RuntimeError, match="render-only"):
            operation()
    _report(
        tmp_path,
        "archival-boundary-report.json",
        {
            "passed": True,
            "physics_stepped": False,
            "fixture_kind": "synthetic_schema1_envelope_with_current_mesh_not_historical_compatibility",
            "recorded_geometry": env.recorded_geometry_identity,
        },
    )


def _source_hashes(source):
    from cloth_fling.rendering import file_sha256

    root = source if source.is_dir() else source.parent
    return {
        str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.mark.parametrize("source_kind", ["DATASET", "DIAGNOSTIC"])
def test_bounded_historical_native_replay_source_hashes(source_kind, tmp_path):
    variable = f"CLOTH_FLING_GENESIS144_REPLAY_{source_kind}"
    if not os.environ.get(variable):
        pytest.skip(f"set {variable} to an explicit historical source after lease grant")
    from cloth_fling.rendering import file_sha256
    from cloth_fling.replay import load_replay, render_replay
    from cloth_fling.sim.backend import GenesisBackend

    source = Path(os.environ[variable]).resolve(strict=True)
    before = _source_hashes(source)
    actions, origin = load_replay(source)
    assert all(a.state["schema_version"] == 1 for a in actions), (
        "expected historical schema-1 source"
    )
    report = {
        "passed": False,
        "source_kind": source_kind,
        "source": str(source),
        "source_files_sha256": before,
        "inspection": origin,
        "max_frames": 2,
    }
    _report(tmp_path, "historical-replay-report.json", report)

    class NoStepBackend(GenesisBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scene.step = self.scene.reset = _forbidden

        step = reset = _forbidden

    # Keep the production template-selection path (backend_factory=None).
    from unittest.mock import patch

    try:
        with patch("cloth_fling.sim.backend.GenesisBackend", NoStepBackend):
            metadata = render_replay(source, tmp_path / "rendered", max_frames=2, fps=10)
        assert metadata["physics_reexecuted"] is False
        assert metadata["original_video_recorded"] is False
        assert all(v["frames"] <= 2 for v in metadata["videos"])
        assert all(e["version"] == "0.4.3" for e in metadata["source_engines"])
        assert all(e["version"] == "1.4.4" for e in metadata["renderer_engines"])
        assert metadata["runtime_qualification_inherited"] is False
        assert all(
            file_sha256(tmp_path / "rendered" / v["path"]) == v["sha256"]
            for v in metadata["videos"]
        )
        report["replay_provenance"] = metadata
        report["passed"] = True
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        after = _source_hashes(source)
        report["source_unchanged"] = after == before
        _report(tmp_path, "historical-replay-report.json", report)
        assert after == before, "historical source file set or hashes changed"
