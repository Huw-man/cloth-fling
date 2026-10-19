"""Exact PBD physics-triangle rendering adapter for Genesis 0.4.3 only.

The pinned private API is deliberately isolated here. Before scene.build(),
replace only the visual topology with a detached copy of the remeshed cloth.
After build, replace only visual support indices/weights with identity supports.
No Genesis/torch imports or initialization occur when importing this module.
"""

from __future__ import annotations

import numpy as np


def _install_two_sided_depth_program(renderer, flags):
    """Keep native depth-only sampling while supplying native two-sided triangles.

    0.4.3 depth-only rendering forces backface culling and omits the RGB geometry
    shader. Its segmentation shader uses the identical position transform plus
    the two-sided geometry shader. Reuse that program for double-sided depth;
    its color output is ignored by the depth-only framebuffer readback. This is
    camera-renderer-local, not a module/global renderer patch.
    """
    original = renderer._get_primitive_program

    def get_program(primitive, render_flags, program_flags):
        if render_flags & flags.DEPTH_ONLY and primitive.double_sided:
            render_flags = (render_flags & ~flags.DEPTH_ONLY) | flags.SEG
        return original(primitive, render_flags, program_flags)

    renderer._get_primitive_program = get_program


class PhysicsMeshRendering:
    """Single unbatched PBD cloth; fail closed when the pinned layout changes."""

    def __init__(self, scene, cloth, genesis):
        if genesis.__version__ != "0.4.3":
            raise RuntimeError("physics-mesh rendering requires Genesis 0.4.3")
        if scene.is_built:
            raise RuntimeError("visual topology must be installed before scene.build")
        solver = scene.pbd_solver
        if (len(solver.entities) != 1 or solver.entities[0] is not cloth
                or type(cloth).__name__ != "PBD2DEntity"
                or cloth.particle_start != 0 or cloth.vvert_start != 0
                or cloth.vface_start != 0 or cloth.surface.vis_mode != "visual"):
            raise RuntimeError("physics-mesh rendering requires one visual PBD cloth at offset zero")
        self.scene, self.cloth, self.solver = scene, cloth, solver
        self._physics_mesh = cloth.mesh
        self._positions = np.asarray(cloth.mesh.verts).copy()
        self._faces = np.asarray(cloth.mesh.faces).copy()
        self._particles = cloth._particles.copy()
        if (self._positions.shape != (cloth.n_particles, 3)
                or self._faces.ndim != 2 or self._faces.shape[1] != 3 or len(self._faces) == 0
                or not np.isfinite(self._positions).all()
                or not np.array_equal(self._positions.astype(genesis.np_float), self._particles)
                or self._faces.min() < 0 or self._faces.max() >= cloth.n_particles):
            raise RuntimeError("unexpected remeshed cloth particle/topology layout")
        # from_attrs(process=False) preserves the particle order and triangle IDs.
        # The physics mesh, particles, constraints, mass, and material stay intact.
        visual = genesis.Mesh.from_attrs(
            verts=self._positions.copy(), faces=self._faces.copy(),
            surface=cloth.surface.model_copy())
        if (visual is cloth.mesh or np.shares_memory(visual.verts, cloth.mesh.verts)
                or np.shares_memory(visual.faces, cloth.mesh.faces)):
            raise RuntimeError("visual mesh must not alias physics topology")
        cloth._vmesh = visual
        cloth._vverts = np.asarray(visual.verts, dtype=genesis.np_float).copy()
        cloth._vfaces = np.asarray(visual.faces, dtype=genesis.np_int).copy()
        self._assert_physics_topology_unchanged()

    def _assert_physics_topology_unchanged(self):
        if (self.cloth.mesh is not self._physics_mesh
                or not np.array_equal(self.cloth.mesh.verts, self._positions)
                or not np.array_equal(self.cloth.mesh.faces, self._faces)
                or not np.array_equal(self.cloth._particles, self._particles)):
            raise RuntimeError("visual adapter changed physics topology or initial particles")

    def _state_arrays(self):
        state = self.solver.get_state(self.scene.sim.cur_substep_local)
        return tuple(getattr(state, name).detach().cpu().numpy().copy()
                     for name in ("pos", "vel", "free"))

    def install_identity_supports(self):
        """Write render-only fields after allocation; verify state and device readback."""
        cloth, solver = self.cloth, self.solver
        if not self.scene.is_built:
            raise RuntimeError("identity supports require a built scene")
        n, faces = cloth.n_particles, len(self._faces)
        if (self.scene.n_envs != 0 or solver._B != 1 or len(solver.entities) != 1
                or solver.entities[0] is not cloth or solver._n_particles != n
                or solver._n_vverts != n or solver._n_vfaces != faces
                or cloth.n_vverts != n or cloth.n_vfaces != faces
                or solver._n_vvert_supports < 1
                or not np.array_equal(cloth._vfaces, self._faces)
                or not np.array_equal(cloth.vmesh.faces, self._faces)):
            raise RuntimeError("unexpected built PBD rendering field layout")
        before = self._state_arrays()
        # Repeated own-particle indices keep every zero-weight support in bounds.
        indices = np.repeat(np.arange(n, dtype=np.int32)[:, None],
                            solver._n_vvert_supports, axis=1)
        weights = np.zeros(indices.shape, dtype=np.float32)
        weights[:, 0] = 1
        solver.vverts_info.support_idxs.from_numpy(indices)
        solver.vverts_info.support_weights.from_numpy(weights)
        if (not np.array_equal(solver.vverts_info.support_idxs.to_numpy(), indices)
                or not np.array_equal(solver.vverts_info.support_weights.to_numpy(), weights)
                or not np.array_equal(solver.vfaces_indices.to_numpy(), self._faces)):
            raise RuntimeError("identity render support/topology readback failed")
        solver.update_render_fields()
        rendered = solver.vverts_render.pos.to_numpy()
        if rendered.shape != (n, 1, 3) or not np.array_equal(rendered[:, 0], before[0][0]):
            raise RuntimeError("visual vertices do not equal physics particles")
        self._assert_physics_topology_unchanged()
        if any(not np.array_equal(a, b) for a, b in zip(before, self._state_arrays(), strict=True)):
            raise RuntimeError("identity render support installation changed physics state")

    def install_two_sided_depth(self, camera):
        """Enable calibrated centre-ray depth without dropping folded backfaces."""
        if not self.scene.is_built:
            raise RuntimeError("depth adapter requires a built scene")
        from genesis.ext.pyrender import RenderFlags

        target = camera._rasterizer._camera_targets[camera.uid]
        if (type(target).__module__ != "genesis.ext.pyrender.renderer"
                or type(target).__name__ != "Renderer"
                or not callable(getattr(target, "_get_primitive_program", None))):
            raise RuntimeError("unexpected Genesis 0.4.3 camera rasterizer layout")
        before = self._state_arrays()
        _install_two_sided_depth_program(target, RenderFlags)
        if any(not np.array_equal(a, b) for a, b in zip(before, self._state_arrays(), strict=True)):
            raise RuntimeError("depth program installation changed physics state")
