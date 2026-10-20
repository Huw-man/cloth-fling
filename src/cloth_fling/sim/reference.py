"""Genesis reference towel, single-vertex hard pins and explicit split contact."""

from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np

from cloth_fling.contracts import ReferenceSceneConfig
from cloth_fling.reference_geometry import reference_mesh, triangle_area
from cloth_fling.sim.backend import ENGINE_VERSION, GenesisBackend, _json
from cloth_fling.sim.coverage import CoverageGrid
from cloth_fling.sim.mesh import mesh_edges, mesh_identity, visible_particle, write_obj
from cloth_fling.sim.render_mesh import PhysicsMeshRendering


def split_contact(positions, before, after, *, free, bounds, support_z, dt, mu=0.45):
    """Post-native normal projection; never changes position or normal velocity."""
    p, vb, va = map(lambda x: np.asarray(x, dtype=float), (positions, before, after))
    free = np.asarray(free, dtype=bool)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or vb.shape != p.shape
        or va.shape != p.shape
        or free.shape != (len(p),)
        or not all(np.isfinite(x).all() for x in (p, vb, va))
        or not np.isfinite([*bounds, support_z, dt, mu]).all()
        or dt <= 0
        or mu < 0
    ):
        raise ValueError("invalid split-contact inputs")
    xmin, xmax, ymin, ymax = bounds
    support = (
        free
        & (np.abs(p[:, 2] - support_z) <= 1e-6)
        & (p[:, 0] >= xmin)
        & (p[:, 0] <= xmax)
        & (p[:, 1] >= ymin)
        & (p[:, 1] <= ymax)
    )
    normal = np.maximum(va[:, 2] - vb[:, 2] + 9.81 * dt, 0)
    speed = np.linalg.norm(va[:, :2], axis=1)
    reduction = np.where(support, np.minimum(speed, mu * normal), 0)
    result = va.copy()
    result[:, :2] *= (1 - np.divide(reduction, speed, out=np.zeros_like(speed), where=speed > 0))[
        :, None
    ]
    ids = np.flatnonzero(reduction > 0)
    return result, ids


class ReferenceBackend(GenesisBackend):
    """Reuses snapshot/rendering codec, with an independent physical configuration."""

    def __init__(self, config=None, *, backend="gpu", headless=True, appearance=None):
        config = config or ReferenceSceneConfig()
        if not isinstance(config, ReferenceSceneConfig) or backend not in ("gpu", "cpu"):
            raise ValueError("reference backend requires ReferenceSceneConfig and gpu/cpu")
        self.config, self.appearance = config, appearance
        self.substeps = config.substeps
        self.simulated_time = 0.0
        self.attached_ids = ()
        self._targets = self._observation = None
        self._closed = self._render_buffers_ready = False
        self.reset_metadata = {"kind": "authored", "qualified": False}
        self._config_json = _json(config.to_dict())
        self._config_hash = hashlib.sha256(self._config_json.encode()).hexdigest()
        self.contact_accounting = {
            "corrected_particles": 0,
            "steps": 0,
            "tangential_speed_removed_m_s": 0.0,
        }
        self.tracking_error_m = 0.0
        self.release_evidence = {}
        if headless:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import genesis as gs

        self._gs = gs
        requested = gs.cuda if backend == "gpu" else gs.cpu
        if gs.__version__ != ENGINE_VERSION:
            raise RuntimeError(f"reference requires Genesis {ENGINE_VERSION}")
        if not gs._initialized:
            gs.init(backend=requested, precision="32", seed=config.seed, logging_level="warning")
        if gs.backend != requested or (backend == "gpu" and gs.device.type != "cuda"):
            raise RuntimeError("reference runtime backend mismatch")
        if gs.np_float != np.float32:
            raise RuntimeError("reference requires 32-bit precision")
        self._runtime_settings = {
            "array_mode": "ndarray" if gs.use_ndarray else "field",
            "deterministic_algorithms": bool(gs.use_deterministic_algorithms),
        }
        self._temporary = tempfile.TemporaryDirectory(prefix="cloth-reference-")
        self.scene = None
        try:
            rest, authored_faces, self.xy_contraction = reference_mesh(config)
            self.authored_positions, self.authored_faces = rest.copy(), authored_faces.copy()
            path = os.path.join(self._temporary.name, "towel.obj")
            write_obj(path, rest, authored_faces)
            self.particle_diameter = config.mesh_spacing_m
            self._initial_height = config.cloth_center_m[2]
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt=config.sim_dt, substeps=config.substeps, gravity=(0, 0, -9.81)
                ),
                rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
                coupler_options=gs.options.LegacyCouplerOptions(),
                pbd_options=gs.options.PBDOptions(
                    max_stretch_solver_iterations=config.max_stretch_solver_iterations,
                    max_bending_solver_iterations=config.max_bending_solver_iterations,
                    particle_size=self.particle_diameter,
                    lower_bound=(-1.2, -1.2, -0.1),
                    upper_bound=(1.2, 1.2, 2.0),
                ),
                show_viewer=not headless,
            )
            from cloth_fling.sim.appearance import make_surface

            table = self.scene.add_entity(
                gs.morphs.Box(
                    size=(*config.table_dimensions_m, config.table_thickness_m),
                    pos=(
                        *config.table_center_xy_m,
                        config.table_surface_z_m - config.table_thickness_m / 2,
                    ),
                    fixed=True,
                ),
                surface=make_surface(gs, appearance)
                if appearance
                else gs.surfaces.Default(color=(0.04, 0.04, 0.04, 1)),
            )
            if appearance:
                from cloth_fling.sim.appearance import install_visual_texture

                install_visual_texture(self.scene, table, gs, appearance)
            self.cloth = self.scene.add_entity(
                gs.morphs.Mesh(file=path, pos=config.cloth_center_m),
                material=gs.materials.PBD.Cloth(
                    rho=config.areal_density_kg_m2,
                    stretch_compliance=config.stretch_compliance,
                    bending_compliance=config.bending_compliance,
                    stretch_relaxation=config.stretch_relaxation,
                    bending_relaxation=config.bending_relaxation,
                    static_friction=config.static_friction,
                    kinetic_friction=config.kinetic_friction,
                    air_resistance=config.air_resistance,
                ),
                surface=gs.surfaces.Default(color=(0.85, 0.25, 0.12, 1), double_sided=True),
            )
            self.camera = self.scene.add_camera(
                res=(config.image_size, config.image_size),
                pos=(0.55, 0, 1.8),
                lookat=(0.55, 0, 0.30),
                up=(0, 1, 0),
                fov=45,
                GUI=False,
            )
            self._physics_mesh_rendering = PhysicsMeshRendering(self.scene, self.cloth, gs)
            self.scene.build()
            if appearance:
                from cloth_fling.sim.appearance import install_native_material

                install_native_material(self.scene, table, gs, appearance)
            self._physics_mesh_rendering.install_identity_supports()
            self._physics_mesh_rendering.install_two_sided_depth(self.camera)
            self._authored_native = self.get_positions()
            self.rest_positions = self._authored_native - np.asarray(config.cloth_center_m)
            self.faces = np.asarray(self.cloth.mesh.faces, dtype=np.int32).copy()
            if not np.isfinite(self.rest_positions).all() or not len(self.faces):
                raise RuntimeError("invalid native reference topology")
            self.native_rest_area_m2 = triangle_area(self.rest_positions, self.faces)
            self.edges = mesh_edges(self.faces)
            self.rest_edge_lengths = np.linalg.norm(
                self.rest_positions[self.edges[:, 1]] - self.rest_positions[self.edges[:, 0]],
                axis=1,
            )
            self._mesh_hash = mesh_identity(self.rest_positions, self.faces)
            self._grid = CoverageGrid(
                cell_size=config.coverage_cell_m, bounds=config.coverage_bounds_m
            )
            flat = self.rest_positions.copy()
            flat[:, :2] /= self.xy_contraction
            flat += np.asarray(config.cloth_center_m)
            self._flat_area = self._grid.area(flat, self.faces)
            pose = np.asarray(self.camera.transform, dtype=float).copy()
            pose[:3, 1:3] *= -1
            self.calibration = {
                "model": "perspective",
                "intrinsics": self.camera.intrinsics.tolist(),
                "camera_to_world": pose.tolist(),
                "image_size": [config.image_size] * 2,
                "pixel_center_offset": 0.5,
                "depth_units": "metres",
                "depth_convention": "positive_camera_z",
                "world_up": "+z",
            }
        except BaseException:
            self.close()
            raise

    @property
    def runtime_identity(self):
        return {
            "engine": "genesis",
            "engine_version": ENGINE_VERSION,
            "physics_schema_version": 2,
            "solver": "PBD",
            "coupler": "LegacyCoupler+reference_split_v1",
            "contact_profile": self.config.contact_profile,
            "profile": self.config.profile,
            "sim_dt": self.config.sim_dt,
            "substeps": self.substeps,
            "max_stretch_solver_iterations": self.config.max_stretch_solver_iterations,
            "precision": "32",
            **self._runtime_settings,
        }

    @property
    def capabilities(self):
        return {
            **super().capabilities,
            "native_contact_accounting": False,
            "contact_qualification": "reference_unqualified",
            "attachment": "single_vertex",
        }

    def take_contact_accounting(self, *, reset=False):
        result = {
            "schema_version": 1,
            "profile": self.config.contact_profile,
            "available": True,
            **self.contact_accounting,
        }
        if reset:
            for key in self.contact_accounting:
                self.contact_accounting[key] = 0
        return result

    def attach(self, pixels):
        self._ensure_physics()
        if self.attached_ids:
            raise ValueError("release before attaching")
        positions = self.get_positions()
        ids = tuple(
            visible_particle(
                p, positions, self.faces, self.calibration, self.particle_diameter * 1.5
            )
            for p in pixels
        )
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError("requires two distinct visible vertices")
        span = np.linalg.norm(positions[ids[0]] - positions[ids[1]])
        xmin, xmax, ymin, ymax = self.config.table_bounds_m
        points = positions[list(ids)]
        if (
            not self.config.min_grasp_separation_m <= span <= self.config.max_grasp_separation_m
            or not (
                (points[:, 0] >= xmin)
                & (points[:, 0] <= xmax)
                & (points[:, 1] >= ymin)
                & (points[:, 1] <= ymax)
            ).all()
        ):
            raise ValueError("grasp outside feasible workspace")
        # Journal before the native mutation, so cleanup includes a partially failed fix.
        self.attached_ids = ids
        self._targets = points.copy()
        self.tracking_error_m = 0.0
        self.release_evidence = {}
        self.cloth.fix_particles(list(ids), zero_velocity=False)
        if self._free()[list(ids)].any():
            raise RuntimeError("native ownership activation failed")
        self._dirty()
        return ids

    def set_attachment_targets(self, targets):
        self._ensure_physics()
        targets = np.asarray(targets, dtype=float)
        if (
            targets.shape != (2, 3)
            or len(self.attached_ids) != 2
            or not np.isfinite(targets).all()
            or (np.abs(targets[:, :2]) >= 1.2).any()
            or (targets[:, 2] < self.config.table_surface_z_m).any()
            or (targets[:, 2] > 1.5).any()
        ):
            raise ValueError("invalid reference TCP targets")
        self._targets = targets.copy()

    def step(self, count=1):
        self._ensure_physics()
        if type(count) is not int or count < 0:
            raise ValueError("step count must be a nonnegative integer")
        for _ in range(count):
            if self.attached_ids:
                positions = self.get_positions()
                self.cloth.set_particles_vel(
                    (self._targets - positions[list(self.attached_ids)]) / self.config.sim_dt,
                    list(self.attached_ids),
                )
            before = self.get_velocities()
            self.scene.step()
            positions, after = self.get_positions(), self.get_velocities()
            free = np.ones(len(positions), dtype=bool)
            free[list(self.attached_ids)] = False
            corrected, ids = split_contact(
                positions,
                before,
                after,
                free=free,
                bounds=self.config.table_bounds_m,
                support_z=self.config.table_surface_z_m + self.particle_diameter / 2,
                dt=self.config.sim_dt,
                mu=np.sqrt(self.config.kinetic_friction * self.config.table_coefficients[1])
                if self.config.table_friction
                else 0,
            )
            if len(ids):
                self.cloth.set_particles_vel(corrected[ids], ids.tolist())
            self.contact_accounting["steps"] += 1
            self.contact_accounting["corrected_particles"] += len(ids)
            self.contact_accounting["tangential_speed_removed_m_s"] += float(
                (
                    np.linalg.norm(after[:, :2], axis=1) - np.linalg.norm(corrected[:, :2], axis=1)
                ).sum()
            )
            self.simulated_time += self.config.sim_dt
            self._dirty()
            if self.attached_ids:
                error = float(
                    np.linalg.norm(positions[list(self.attached_ids)] - self._targets, axis=1).max()
                )
                self.tracking_error_m = max(self.tracking_error_m, error)
                if error > 1e-6:
                    raise RuntimeError("hard-pin tracking tolerance exceeded")

    def release(self):
        self._ensure_open()
        if not self.attached_ids:
            return
        ids = list(self.attached_ids)
        velocity = self.get_velocities()[ids]
        self.cloth.release_particle(ids)
        self.cloth.set_particles_vel(velocity, ids)
        error = float(np.abs(self.get_velocities()[ids] - velocity).max())
        free = self._free()
        self.release_evidence = {
            "vertex_ids": ids,
            "velocities_before_m_s": velocity,
            "velocity_error_m_s": error,
            "native_free": free[ids].copy(),
            "passed": bool(free.all() and error <= 1e-4),
        }
        if not self.release_evidence["passed"]:
            raise RuntimeError("native release verification failed")
        self.attached_ids, self._targets = (), None
        self._dirty()

    def reset(self, seed=0, *, kind="authored"):
        self._ensure_physics()
        if kind != "authored":
            raise ValueError("reference profile supports authored reset only")
        self.release()
        self.scene.reset()
        self.cloth.release_particle()
        self.cloth.set_particles_vel(np.zeros_like(self._authored_native))
        self.simulated_time = 0.0
        self.reset_metadata = {
            "kind": "authored",
            "seed": int(seed),
            "qualified": False,
            "native_rest_area_m2": self.native_rest_area_m2,
            "authored_area_m2": self.config.cloth_size_m * self.config.cloth_width_m,
            "xy_contraction": self.xy_contraction,
        }
        self._dirty()
        return self.observe()
