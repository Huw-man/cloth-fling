"""Offline checks of the pinned rendering adapter's layout and isolation guards."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_fling.sim.render_mesh import PhysicsMeshRendering, _install_two_sided_depth_program


class Field:
    def __init__(self, value):
        self.value = np.asarray(value).copy()

    def from_numpy(self, value):
        self.value[:] = value

    def to_numpy(self):
        return self.value.copy()


class Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Surface:
    vis_mode = "visual"
    double_sided = True

    def model_copy(self):
        return Surface()


class Mesh:
    def __init__(self, verts, faces, surface):
        self.verts, self.faces = verts.copy(), faces.copy()
        self.surface = surface

    @staticmethod
    def from_attrs(**kwargs):
        return Mesh(**kwargs)


class PBD2DEntity:
    particle_start = vvert_start = vface_start = 0

    @property
    def vmesh(self):
        return self._vmesh

    @property
    def n_vverts(self):
        return len(self._vverts)

    @property
    def n_vfaces(self):
        return len(self._vfaces)


@pytest.fixture
def layout():
    surface = Surface()
    p = np.array(
        [[0, 0, 0.006], [0.01, 0, 0.006], [0.01, 0.01, 0.006], [0, 0.01, 0.006]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    cloth = PBD2DEntity()
    cloth.mesh = Mesh(p, faces, surface)
    cloth._particles = p.copy()
    cloth.n_particles = len(p)
    cloth.surface = surface
    state = SimpleNamespace(
        pos=Tensor(p[None].copy()),
        vel=Tensor(np.ones((1, 4, 3))),
        free=Tensor(np.ones((1, 4), dtype=bool)),
    )
    solver = SimpleNamespace(
        entities=[cloth],
        _B=1,
        _n_particles=4,
        _n_vverts=4,
        _n_vfaces=2,
        _n_vvert_supports=4,
        vverts_info=SimpleNamespace(
            support_idxs=Field(np.zeros((4, 4), dtype=np.int32)),
            support_weights=Field(np.full((4, 4), 0.25)),
        ),
        vfaces_indices=Field(faces),
        vverts_render=SimpleNamespace(pos=Field(np.zeros((4, 1, 3)))),
        get_state=lambda _: state,
    )

    def update_render_fields():
        index = solver.vverts_info.support_idxs.to_numpy()
        weight = solver.vverts_info.support_weights.to_numpy()
        solver.vverts_render.pos.from_numpy(
            np.sum(state.pos.value[0][index] * weight[:, :, None], axis=1)[:, None]
        )

    solver.update_render_fields = update_render_fields
    scene = SimpleNamespace(
        is_built=False, pbd_solver=solver, n_envs=0, sim=SimpleNamespace(cur_substep_local=0)
    )
    gs = SimpleNamespace(__version__="1.4.4", Mesh=Mesh, np_float=np.float32, np_int=np.int32)
    return scene, cloth, gs, state


def test_visual_copy_identity_supports_and_unchanged_physics(layout):
    scene, cloth, gs, state = layout
    before = [
        x.copy()
        for x in (
            cloth.mesh.verts,
            cloth.mesh.faces,
            cloth._particles,
            state.pos.value,
            state.vel.value,
            state.free.value,
        )
    ]
    adapter = PhysicsMeshRendering(scene, cloth, gs)
    assert cloth.vmesh is not cloth.mesh
    assert not np.shares_memory(cloth.vmesh.verts, cloth.mesh.verts)
    assert not np.shares_memory(cloth.vmesh.faces, cloth.mesh.faces)
    scene.is_built = True
    adapter.install_identity_supports()
    for a, b in zip(
        before,
        (
            cloth.mesh.verts,
            cloth.mesh.faces,
            cloth._particles,
            state.pos.value,
            state.vel.value,
            state.free.value,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    weights = scene.pbd_solver.vverts_info.support_weights.to_numpy()
    np.testing.assert_array_equal(weights[:, 0], 1)
    np.testing.assert_array_equal(weights[:, 1:], 0)
    # A nonplanar test fixture follows exactly through identity visual supports.
    state.pos.value[0, :, 2] += [0, 0.1, 0.2, 0.05]
    scene.pbd_solver.update_render_fields()
    np.testing.assert_array_equal(
        scene.pbd_solver.vverts_render.pos.to_numpy()[:, 0], state.pos.value[0]
    )


def test_metadata_distinguishes_installed_mechanisms_from_qualification(layout, monkeypatch):
    scene, cloth, gs, _ = layout
    adapter = PhysicsMeshRendering(scene, cloth, gs)
    assert adapter.identity["engine_version"] == "1.4.4"
    assert adapter.identity["native_renderer_classes"] is None
    assert adapter.identity["visual_supports"] is None
    assert not adapter.capabilities["identity_visual_supports_installed"]
    scene.is_built = True
    adapter.install_identity_supports()
    assert adapter.identity["visual_supports"] == "identity"
    assert adapter.capabilities["identity_visual_supports_installed"]

    class Renderer:
        def _get_primitive_program(self, primitive, flags, program_flags):
            return "test_program"

    Renderer.__module__ = "genesis.ext.pyrender.renderer"
    target = Renderer()
    camera = SimpleNamespace(
        uid="test", _rasterizer=SimpleNamespace(_camera_targets={"test": target})
    )
    monkeypatch.setitem(
        sys.modules,
        "genesis.ext.pyrender",
        SimpleNamespace(RenderFlags=SimpleNamespace(DEPTH_ONLY=1, SEG=2)),
    )
    # Native source dispatch is independently audited; exercise installation and
    # metadata without importing Genesis or creating a graphics context.
    monkeypatch.setattr(
        "cloth_fling.sim.render_mesh.inspect.getsource",
        lambda _: (
            """
not flags & RenderFlags.DEPTH_ONLY
elif flags & RenderFlags.SEG:
geometry_shader = "segmentation_double_sided.geom"
vertex_shader = "mesh_depth.vert"
"""
        ),
    )
    adapter.install_two_sided_depth(camera)
    assert adapter.identity["native_renderer_classes"] == ["genesis.ext.pyrender.renderer.Renderer"]
    assert adapter.capabilities["double_sided_depth_camera_count"] == 1
    for key, value in adapter.capabilities.items():
        if key.endswith("_qualified") or key == "qualification_inherited":
            assert value is False
    assert adapter.capabilities["qualification_status"] == "unqualified"
    json.dumps(
        {"identity": adapter.identity, "capabilities": adapter.capabilities}, allow_nan=False
    )
    modified = adapter.identity
    modified["native_renderer_classes"].clear()
    assert adapter.identity["native_renderer_classes"]
    modified = adapter.capabilities
    modified["native_gpu_qualified"] = True
    assert adapter.capabilities["native_gpu_qualified"] is False


@pytest.mark.parametrize(
    "fault", ["version", "built", "multiple", "offset", "alias", "one_sided", "reordered"]
)
def test_reject_unsupported_prebuild_layout(layout, fault):
    scene, cloth, gs, _ = layout
    if fault == "version":
        gs.__version__ = "1.4.3"
    elif fault == "built":
        scene.is_built = True
    elif fault == "multiple":
        scene.pbd_solver.entities.append(object())
    elif fault == "offset":
        cloth.vvert_start = 1
    elif fault == "alias":
        gs.Mesh = SimpleNamespace(from_attrs=lambda **_: cloth.mesh)
    elif fault == "one_sided":
        cloth.surface.double_sided = False
    else:
        gs.Mesh = SimpleNamespace(
            from_attrs=lambda **kw: Mesh(kw["verts"][::-1], kw["faces"], kw["surface"])
        )
    with pytest.raises(RuntimeError):
        PhysicsMeshRendering(scene, cloth, gs)


@pytest.mark.parametrize("fault", ["not_built", "batched", "count", "faces", "support_shape"])
def test_reject_unsupported_built_layout(layout, fault):
    scene, cloth, gs, _ = layout
    adapter = PhysicsMeshRendering(scene, cloth, gs)
    scene.is_built = fault != "not_built"
    if fault == "batched":
        scene.n_envs = 1
    elif fault == "count":
        scene.pbd_solver._n_vverts += 1
    elif fault == "faces":
        cloth._vfaces[0, 0] = 1
    elif fault == "support_shape":
        scene.pbd_solver.vverts_info.support_idxs.value = np.zeros((4, 2))
    with pytest.raises(RuntimeError):
        adapter.install_identity_supports()
    assert adapter.capabilities["identity_visual_supports_installed"] is False


def test_depth_program_change_is_local_and_only_for_double_sided_depth():
    calls = []
    flags = SimpleNamespace(DEPTH_ONLY=1, SEG=2)

    def original(primitive, render_flags, program_flags):
        calls.append((primitive.double_sided, render_flags, program_flags))
        return "native_program"

    renderer = SimpleNamespace(_get_primitive_program=original)
    _install_two_sided_depth_program(renderer, flags)
    for double_sided, render_flags in [(True, 1), (False, 1), (True, 0), (True, 5)]:
        assert (
            renderer._get_primitive_program(
                SimpleNamespace(double_sided=double_sided), render_flags, 99
            )
            == "native_program"
        )
    assert calls == [(True, 2, 99), (False, 1, 99), (True, 0, 99), (True, 6, 99)]
    assert original is not renderer._get_primitive_program
