"""CPU checks for privileged rendering isolation and optional encoder lifecycle."""

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_fling.contracts import SceneConfig
from cloth_fling.rendering import (
    FrameWriter,
    VideoRecorder,
    debug_markers,
    marker_positions,
    preserved_camera,
    render_debug_views,
)
from cloth_fling.sim.appearance import (
    AppearanceConfig,
    install_native_material,
    install_visual_texture,
    make_surface,
)


class DebugScene:
    def __init__(self):
        self.objects = {"unrelated": None}
        self.drawn = []
        self.fail_draw = False

    def draw_debug_sphere(self, **kwargs):
        if self.fail_draw and self.drawn:
            raise RuntimeError("draw failed")
        key = f"sphere-{len(self.drawn)}"
        self.objects[key] = kwargs
        self.drawn.append(kwargs)
        return key

    def clear_debug_object(self, key):
        del self.objects[key]


class Camera:
    def __init__(self, scene):
        self.scene = scene
        self.transform = np.eye(4)
        self._lookat = np.array([0.0, 0.0, -1.8])
        self._debug = False
        self.calls = 0
        self.fail_at = None

    def set_pose(self, transform=None, **kwargs):
        if transform is None:
            self.transform = np.eye(4) * 2
            self._lookat[:] = kwargs["lookat"]
        else:
            self.transform = transform.copy()
            self._lookat[:] = [0, 0, -1]

    def render(self, **kwargs):
        assert kwargs == dict(rgb=True, depth=False, antialiasing=False, force_render=True)
        assert self._debug is True
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("render failed")
        return np.full((16, 16, 3), len(self.scene.objects), dtype=np.uint8), None, None, None


class Env:
    def __init__(self):
        self.scene = DebugScene()
        self.camera = Camera(self.scene)
        self.p = np.array([[0.0, 0.0, 0.1], [0.2, 0.0, 0.2], [0.1, 0.1, 0.01]])
        self.v = np.ones_like(self.p)
        self.attached_ids = (0, 1)
        self.config = SceneConfig(image_size=16)
        self._gs = SimpleNamespace(__version__="1.4.4")
        self.runtime_identity = {"genesis_version": "1.4.4", "snapshot_schema_version": 2}
        self.simulated_time = 3.0
        self._observation = SimpleNamespace(rgb=np.ones((16, 16, 3)), depth=np.ones((16, 16)))
        self._targets = np.zeros((2, 3))
        self._render_buffers_ready = False

    def get_positions(self):
        return self.p.copy()

    def observe(self):
        raise AssertionError("debug renderer must not call observe")


class Encoder:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.frames = []
        self.closed = 0

    def append_data(self, frame):
        self.frames.append(frame)
        self.path.write_bytes(b"fake encoded video")

    def close(self):
        self.closed += 1


def test_measured_markers_and_immediate_release_hiding():
    env = Env()
    points = marker_positions(env.p, (1, 0), "lift", np.ones((2, 3)) * 9)
    np.testing.assert_array_equal(points, env.p[[1, 0]])
    for phase in ("release", "released", "settle", "terminal"):
        assert marker_positions(env.p, (), phase, env.p[:2]) is None
    np.testing.assert_array_equal(marker_positions(env.p, (), "pre_attach", env.p[:2]), env.p[:2])
    with pytest.raises(ValueError):
        marker_positions(env.p, (-1, 0), "lift")


def test_batched_native_pose_fields_and_debug_flag_restore_even_when_set_pose_fails():
    camera = Camera(DebugScene())
    for name, shape in (
        ("_pos", (1, 3)),
        ("_up", (1, 3)),
        ("_quat", (1, 4)),
        ("_transform", (1, 4, 4)),
    ):
        setattr(camera, name, np.ones(shape))
    saved = {k: v.copy() for k, v in camera.__dict__.items() if isinstance(v, np.ndarray)}
    with pytest.raises(RuntimeError, match="restore failure"):
        with preserved_camera(camera):
            assert camera._debug
            for name in saved:
                if name.startswith("_"):
                    getattr(camera, name)[:] = 2
            camera.set_pose = lambda **kw: (_ for _ in ()).throw(RuntimeError("restore failure"))
    assert not camera._debug
    for name, value in saved.items():
        if name.startswith("_"):
            np.testing.assert_array_equal(getattr(camera, name), value)


@pytest.mark.parametrize("field", ["_batch_renderer", "_raytracer"])
def test_non_raster_camera_fails_before_drawing(field):
    env = Env()
    setattr(env.camera, field, object())
    with pytest.raises(RuntimeError, match="native rasterizer"):
        render_debug_views(env, phase="lift")
    assert not env.scene.drawn


@pytest.mark.parametrize("failure", [None, 1, 3])
def test_native_debug_preserves_camera_state_policy_caches_and_unrelated_objects(failure):
    env = Env()
    original = copy.deepcopy(env.__dict__)
    observation = env._observation
    env.camera.fail_at = failure
    if failure:
        with pytest.raises(RuntimeError, match="render failed"):
            render_debug_views(env, phase="lift")
    else:
        views = render_debug_views(env, phase="lift")
        assert set(views) == {"overhead", "oblique"}
    assert set(env.scene.objects) == {"unrelated"}
    assert env.camera._debug is False
    assert env._observation is observation
    np.testing.assert_array_equal(env._observation.rgb, original["_observation"].rgb)
    np.testing.assert_array_equal(env._observation.depth, original["_observation"].depth)
    np.testing.assert_array_equal(env.camera.transform, original["camera"].transform)
    np.testing.assert_array_equal(env.camera._lookat, original["camera"]._lookat)
    np.testing.assert_array_equal(env.p, original["p"])
    np.testing.assert_array_equal(env.v, original["v"])
    np.testing.assert_array_equal(env._targets, original["_targets"])
    assert env.simulated_time == 3 and env.attached_ids == (0, 1)
    assert not env._render_buffers_ready
    assert [draw["radius"] for draw in env.scene.drawn] == [0.01, 0.01]
    np.testing.assert_array_equal(env.scene.drawn[0]["pos"], env.p[0])


def test_partial_marker_creation_failure_clears_first():
    scene = DebugScene()
    scene.fail_draw = True
    with pytest.raises(RuntimeError):
        with debug_markers(scene, np.zeros((2, 3))):
            pass
    assert set(scene.objects) == {"unrelated"}


def test_framewriter_validates_closes_and_does_not_alias(tmp_path):
    encoders = []

    def factory(*args, **kwargs):
        encoder = Encoder(*args, **kwargs)
        encoders.append(encoder)
        return encoder

    writer = FrameWriter(tmp_path / "video.mp4", fps=15, writer_factory=factory)
    frame = np.zeros((16, 18, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="owner failed"):
        with writer:
            writer.append(frame)
            frame[:] = 255
            with pytest.raises(ValueError, match="shape changed"):
                writer.append(np.zeros((18, 18, 3), dtype=np.uint8))
            raise RuntimeError("owner failed")
    writer.close()
    assert encoders[0].closed == 1
    assert not encoders[0].frames[0].any()
    assert writer.metadata()["frames"] == 1
    assert writer.metadata()["dimensions"] == [18, 16]
    assert writer.metadata()["sha256"]
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(frame)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, 241])
def test_framewriter_fps_validation(tmp_path, fps):
    with pytest.raises(ValueError):
        FrameWriter(tmp_path / "a.mp4", fps=fps)


def test_appearance_validation_deterministic_gray_and_legacy():
    config = AppearanceConfig()
    texture = config.texture_rgb()
    np.testing.assert_array_equal(texture, config.texture_rgb())
    np.testing.assert_array_equal(texture[:, :, 0], texture[:, :, 2])
    assert np.ptp(texture) > 0
    assert np.abs(texture.astype(float) / 255 - 0.55).max() < 0.015
    gs = SimpleNamespace(surfaces=SimpleNamespace(Default=lambda **kwargs: kwargs))
    assert make_surface(gs)["color"] == (0.04, 0.04, 0.04, 1)
    assert make_surface(gs, config)["roughness"] == 0.8
    assert AppearanceConfig.legacy().texture_amplitude == 0
    assert AppearanceConfig(**json.loads(json.dumps(config.to_dict()))) == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color": [1, 2]},
        {"color": [True, 0.5, 0.5]},
        {"roughness": float("nan")},
        {"metallic": -1},
        {"texture_size": True},
        {"texture_size": 3},
        {"texture_seed": -1},
        {"texture_amplitude": 0.1},
        {"mode": "unknown"},
    ],
)
def test_appearance_bad_fields(kwargs):
    with pytest.raises(ValueError):
        AppearanceConfig(**kwargs)


def test_table_native_pbr_only_changes_visual_material():
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1.2, 1.2, 0.04))
    collision = mesh.copy()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    geom = SimpleNamespace(uid="table", vmesh=SimpleNamespace(trimesh=mesh))
    table = SimpleNamespace(vgeoms=[geom])
    scene = SimpleNamespace(is_built=False)
    gs = SimpleNamespace(__version__="1.4.4")
    config = AppearanceConfig()
    install_visual_texture(scene, table, gs, config)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    assert collision.visual.kind != "texture"
    mat = mesh.visual.material
    assert mat.roughnessFactor == 0.8 and mat.metallicFactor == 0
    native = SimpleNamespace(roughnessFactor=0.8, metallicFactor=0, baseColorTexture=object())
    primitive = SimpleNamespace(material=native, texcoord_0=mesh.visual.uv)
    scene.is_built = True
    scene.visualizer = SimpleNamespace(
        context=SimpleNamespace(
            rigid_nodes={"table": SimpleNamespace(mesh=SimpleNamespace(primitives=[primitive]))}
        )
    )
    install_native_material(scene, table, gs, config)
    native.metallicFactor = 0.2
    with pytest.raises(RuntimeError, match="readback"):
        install_native_material(scene, table, gs, config)


def test_recorder_callback_provenance_dimensions_clearance_and_release(tmp_path):
    env = Env()
    with VideoRecorder(
        tmp_path, writer_factory=Encoder, appearance=AppearanceConfig(), particle_radius_m=0.005
    ) as recorder:
        recorder.start_action(1)
        recorder(env, {"phase": "lift", "simulated_seconds": 0.0, "coverage": 0.4})
        assert len(env.scene.drawn) == 2
        env.attached_ids = ()
        recorder(env, {"phase": "released", "simulated_seconds": 0.1, "coverage": 0.3})
        assert len(env.scene.drawn) == 2
    metadata = json.loads((tmp_path / "video-metadata.json").read_text())
    assert metadata["kind"] == "native_render_capture" and metadata["complete"]
    assert metadata["samples"][-1]["coverage"] == 0.3
    assert metadata["samples"][-1]["min_cloth_clearance_m"] == pytest.approx(0.005)
    assert metadata["videos"][-1]["dimensions"] == [768, 384]
    assert all(v["sha256"] and v["physics_config_sha256"] for v in metadata["videos"])
    assert len(metadata["physics_configs"]) == 1


def test_callback_encoder_failure_is_explicit_in_metadata_even_if_owner_catches(tmp_path):
    class BrokenEncoder(Encoder):
        def append_data(self, frame):
            raise RuntimeError("encoder failure")

    with VideoRecorder(tmp_path, writer_factory=BrokenEncoder) as recorder:
        with pytest.raises(RuntimeError, match="encoder failure"):
            recorder(Env(), {"phase": "lift", "simulated_seconds": 0.0, "coverage": 0.4})
    metadata = recorder.metadata()
    assert not metadata["complete"]
    assert metadata["failure"] == "RuntimeError: encoder failure"


@pytest.mark.parametrize("disabled_at_import", [False, True])
def test_disabled_native_offscreen_markers_fail_clearly(monkeypatch, disabled_at_import):
    if disabled_at_import:
        monkeypatch.delenv("GS_DISABLE_OFFSCREEN_MARKERS", raising=False)
        monkeypatch.setitem(
            sys.modules,
            "genesis.ext.pyrender.jit_render",
            SimpleNamespace(_DISABLE_OFFSCREEN_MARKERS=True),
        )
    else:
        # Genesis treats even a value of "0" as enabled presence of the disable flag.
        monkeypatch.setenv("GS_DISABLE_OFFSCREEN_MARKERS", "0")
    env = Env()
    with pytest.raises(RuntimeError, match="offscreen debug markers are disabled"):
        render_debug_views(env, phase="lift")
    assert not env.scene.drawn and env.camera.calls == 0
    assert set(env.scene.objects) == {"unrelated"}
    assert env.camera._debug is False
