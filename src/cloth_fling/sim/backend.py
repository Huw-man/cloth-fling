"""One-owner Genesis 0.4.3 PBD scene with explicit particle assistance.

Genesis and torch are imported only when constructing a backend. No learned
policy should consume the depth, mesh, calibration, or snapshots exposed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Self

import numpy as np

from cloth_fling.contracts import Observation, SceneConfig
from cloth_fling.sim.coverage import CoverageGrid
from cloth_fling.sim.mesh import mesh_edges, mesh_identity, resolve_grasps, square_mesh, write_obj


class InvalidResetError(RuntimeError):
    """The bounded preparation failed; the caller must record this failure."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _state_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(set(state) - {"state_hash"}):
        value = np.ascontiguousarray(state[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class GenesisBackend:
    """Single cloth, fixed table, overhead perspective RGB/depth camera.

    ``faces``, ``edges``, ``rest_positions`` and ``rest_edge_lengths`` use the
    runtime particle order. ``calibration`` is JSON serializable, with OpenCV
    camera axes (right, down, forward), pixel centres at (u+.5,v+.5).
    ``substeps=4`` is fixed and qualified against the installed solver.
    """

    substeps = 4

    def __init__(self, config: SceneConfig, *, backend: str = "gpu", headless: bool = True):
        self._validate_config(config)
        if backend not in ("gpu", "cpu"):
            raise ValueError("backend must be gpu or cpu")
        self.config = config
        self.simulated_time = 0.0
        self.attached_ids: tuple[int, ...] = ()
        self._targets: np.ndarray | None = None
        self._closed = False
        self._observation: Observation | None = None
        self._render_buffers_ready = False
        self.reset_metadata: dict[str, Any] = {"kind": "initial_flat", "qualified": False}
        self._config_json = _json(config.to_dict())
        self._config_hash = hashlib.sha256(self._config_json.encode()).hexdigest()
        # The headless rasterizer requires EGL, set before importing OpenGL/Genesis.
        if headless:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import genesis as gs

        self._gs = gs
        requested = gs.cuda if backend == "gpu" else gs.cpu
        if not gs._initialized:
            # Request CUDA explicitly: gs.gpu silently permits a CPU fallback.
            gs.init(backend=requested, seed=config.seed, logging_level="warning")
        if gs.backend != requested or (backend == "gpu" and gs.device.type != "cuda"):
            raise RuntimeError("Genesis runtime backend differs from requested backend")
        self._temporary = tempfile.TemporaryDirectory(prefix="cloth-fling-mesh-")
        self.scene = None
        try:
            rest, faces = square_mesh(config.cloth_size_m, config.mesh_spacing_m)
            path = os.path.join(self._temporary.name, "cloth.obj")
            write_obj(path, rest, faces)
            # 0.4.3 ALWAYS isotropically remeshes PBD cloth to particle_size.
            # This is both target edge spacing and collision particle diameter.
            self.particle_diameter = config.mesh_spacing_m
            self._initial_height = self.particle_diameter / 2 + 0.001
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=config.sim_dt, substeps=self.substeps),
                rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
                pbd_options=gs.options.PBDOptions(
                    max_stretch_solver_iterations=config.max_stretch_solver_iterations,
                    particle_size=self.particle_diameter,
                    lower_bound=(-config.table_size_m, -config.table_size_m, -0.1),
                    upper_bound=(config.table_size_m, config.table_size_m, 2.0),
                ),
                show_viewer=not headless,
            )
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(config.table_size_m, config.table_size_m, 0.04),
                    pos=(0, 0, -0.02),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=(0.04, 0.04, 0.04, 1)),
            )
            self.cloth = self.scene.add_entity(
                gs.morphs.Mesh(file=path, pos=(0, 0, self._initial_height)),
                material=gs.materials.PBD.Cloth(
                    rho=config.areal_density_kg_m2,
                    stretch_compliance=config.stretch_compliance,
                    bending_compliance=config.bending_compliance,
                    static_friction=config.static_friction,
                    kinetic_friction=config.kinetic_friction,
                    air_resistance=config.air_resistance,
                ),
                surface=gs.surfaces.Default(color=(0.85, 0.25, 0.12, 1), double_sided=True),
            )
            self.camera = self.scene.add_camera(
                res=(config.image_size, config.image_size),
                pos=(0, 0, 1.8),
                lookat=(0, 0, 0),
                up=(0, 1, 0),
                fov=45,
                GUI=False,
            )
            self.scene.build()
            # Genesis's loader may reorder OBJ vertices. Adopt its topology, then
            # verify the remeshed surface is still the requested rest square.
            self.rest_positions = self.get_positions()
            self.rest_positions[:, 2] -= self._initial_height
            self.faces = np.asarray(self.cloth.mesh.faces, dtype=np.int32).copy()
            triangles = self.rest_positions[self.faces]
            area = (
                np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                    axis=1,
                ).sum()
                / 2
            )
            if (
                not np.allclose(self.rest_positions[:, 2], 0, atol=1e-7)
                or not np.allclose(self.rest_positions.min(axis=0), rest.min(axis=0), atol=1e-7)
                or not np.allclose(self.rest_positions.max(axis=0), rest.max(axis=0), atol=1e-7)
                or not np.isclose(area, config.cloth_size_m**2, atol=1e-6)
            ):
                raise RuntimeError(
                    f"Genesis changed cloth topology: vertices "
                    f"{len(self.rest_positions)}/{len(rest)}, "
                    f"faces {len(self.faces)}/{len(faces)}, bounds "
                    f"{self.rest_positions.min(axis=0)} .. {self.rest_positions.max(axis=0)}"
                )
            self.edges = mesh_edges(self.faces)
            self.rest_edge_lengths = np.linalg.norm(
                self.rest_positions[self.edges[:, 1]] - self.rest_positions[self.edges[:, 0]],
                axis=1,
            )
            self._mesh_hash = mesh_identity(self.rest_positions, self.faces)
            self._grid = CoverageGrid(config.table_size_m)
            self._flat_area = self._grid.area(self.rest_positions, self.faces)
            pose = np.asarray(self.camera.transform, dtype=float).copy()
            pose[:3, 1:3] *= -1  # OpenGL -> OpenCV
            self.calibration = {
                "model": "perspective",
                "intrinsics": self.camera.intrinsics.tolist(),
                "camera_to_world": pose.tolist(),
                "image_size": [config.image_size, config.image_size],
                "pixel_center_offset": 0.5,
                "depth_units": "metres",
                "depth_convention": "positive_camera_z",
                "world_up": "+z",
            }
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validate_config(config: SceneConfig) -> None:
        data = config.to_dict()
        if not all(np.isfinite(value) for value in data.values()):
            raise ValueError("configuration must be finite")
        positive = (
            "cloth_size_m",
            "mesh_spacing_m",
            "areal_density_kg_m2",
            "sim_dt",
            "table_size_m",
            "grasp_workspace_m",
            "max_grasp_separation_m",
        )
        if any(data[key] <= 0 for key in positive):
            raise ValueError("scene dimensions, density and dt must be positive")
        if any(
            data[key] < 0
            for key in (
                "stretch_compliance",
                "bending_compliance",
                "static_friction",
                "kinetic_friction",
                "air_resistance",
            )
        ):
            raise ValueError("material parameters must be nonnegative")
        if not 0 <= config.min_grasp_separation_m <= config.max_grasp_separation_m:
            raise ValueError("invalid grasp separation bounds")
        if (
            config.cloth_size_m > config.table_size_m
            or config.grasp_workspace_m > config.table_size_m
        ):
            raise ValueError("cloth and grasp workspace must fit table")
        if not isinstance(config.image_size, int) or config.image_size < 16:
            raise ValueError("image_size must be an integer >=16")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("backend is closed")

    def _dirty(self) -> None:
        self._observation = None

    def get_positions(self) -> np.ndarray:
        self._ensure_open()
        return self.cloth.get_particles_pos().detach().cpu().numpy().copy()

    def get_velocities(self) -> np.ndarray:
        self._ensure_open()
        return self.cloth.get_particles_vel().detach().cpu().numpy().copy()

    def _free(self) -> np.ndarray:
        state = self.scene.pbd_solver.get_state(self.scene.sim.cur_substep_local)
        return state.free.detach().cpu().numpy().reshape(-1).astype(bool)

    def observe(self) -> Observation:
        self._ensure_open()
        if self._observation is None:
            render_options = {
                "rgb": True,
                "depth": True,
                "antialiasing": False,
                "force_render": True,
            }
            if not self._render_buffers_ready:
                # Genesis 0.4.3 applies pending vertex updates BEFORE the first
                # offscreen draw allocates GL buffers (IDs are still -1). That
                # first draw uploads the original static visual mesh. Discard it
                # and force a second upload of the CURRENT dynamic render fields.
                self.camera.render(**render_options)
                self._render_buffers_ready = True
            # Restore/setters do not increment scene._t. Bypass both visualizer
            # and rasterizer time caches; refreshing rendering must never step.
            rgb, depth, _, _ = self.camera.render(**render_options)
            rgb = np.asarray(rgb, dtype=np.uint8).copy()
            depth = np.asarray(depth, dtype=np.float32).copy()
            if (
                rgb.shape != (self.config.image_size, self.config.image_size, 3)
                or depth.shape != rgb.shape[:2]
                or not np.isfinite(depth).all()
            ):
                raise RuntimeError("invalid rendered RGB/depth observation")
            identity = {
                "positions": self.get_positions(),
                "velocities": self.get_velocities(),
                "free": self._free(),
                "simulated_time": self.simulated_time,
                "config_hash": self._config_hash,
                "mesh_hash": self._mesh_hash,
                "rgb": rgb,
                "depth": depth,
            }
            self._observation = Observation(_state_hash(identity), rgb, depth, self.calibration)
        obs = self._observation
        # A consumer cannot corrupt cached chain identity by modifying an array.
        return Observation(
            obs.observation_id, obs.rgb.copy(), obs.depth.copy(), json.loads(_json(obs.calibration))
        )

    def attach(self, pixels: tuple[tuple[float, float], tuple[float, float]]) -> tuple[int, int]:
        self._ensure_open()
        if self.attached_ids:
            raise ValueError("release existing attachments before attaching")
        ids = resolve_grasps(
            pixels,
            self.get_positions(),
            self.faces,
            self.calibration,
            radius=self.config.mesh_spacing_m * 1.5,
            workspace=self.config.grasp_workspace_m,
            min_separation=self.config.min_grasp_separation_m,
            max_separation=self.config.max_grasp_separation_m,
        )
        self.cloth.fix_particles(list(ids), zero_velocity=False)
        self.attached_ids = ids
        self._targets = self.get_positions()[list(ids)]
        self._dirty()
        return ids

    def set_attachment_targets(self, targets: np.ndarray) -> None:
        self._ensure_open()
        values = np.asarray(targets, dtype=np.float64)
        if not self.attached_ids or values.shape != (len(self.attached_ids), 3):
            raise ValueError("targets must match the active attachment IDs")
        if not np.isfinite(values).all():
            raise ValueError("attachment targets must be finite")
        if (
            np.any(np.abs(values[:, :2]) > self.config.grasp_workspace_m / 2)
            or np.any(values[:, 2] < self.particle_diameter / 2 - 1e-6)
            or np.any(values[:, 2] > 1.5)
        ):
            raise ValueError("attachment target outside safe workspace")
        self._targets = values.copy()

    def step(self, count: int = 1) -> None:
        self._ensure_open()
        if not isinstance(count, int) or count < 0:
            raise ValueError("step count must be a nonnegative integer")
        for _ in range(count):
            if self.attached_ids:
                current = self.get_positions()[list(self.attached_ids)]
                # Four substeps each integrate this velocity for dt/4. No free
                # particle position/velocity writes occur in this stepping path.
                velocity = (self._targets - current) / self.config.sim_dt
                self.cloth.set_particles_vel(velocity, list(self.attached_ids))
            self.scene.step()
            self.simulated_time += self.config.sim_dt
            self._dirty()
            if self.attached_ids:
                actual = self.get_positions()[list(self.attached_ids)]
                if not np.allclose(actual, self._targets, atol=2e-5, rtol=0):
                    raise RuntimeError("attachment endpoint readback failed")

    def release(self) -> None:
        self._ensure_open()
        if self.attached_ids:
            self.cloth.release_particle(list(self.attached_ids))
        self.attached_ids = ()
        self._targets = None
        self._dirty()

    def coverage(self) -> tuple[float, float, float]:
        area = self._grid.area(self.get_positions(), self.faces)
        return area / self._flat_area, area, self._flat_area

    def snapshot(self) -> dict[str, Any]:
        self._ensure_open()
        state = {
            "schema_version": 1,
            "positions": self.get_positions(),
            "velocities": self.get_velocities(),
            "free": self._free(),
            "faces": self.faces.copy(),
            "edges": self.edges.copy(),
            "rest_positions": self.rest_positions.copy(),
            "mesh_hash": self._mesh_hash,
            "config_json": self._config_json,
            "config_hash": self._config_hash,
            "calibration_json": _json(self.calibration),
            "simulated_time": self.simulated_time,
            "reset_metadata_json": _json(self.reset_metadata),
        }
        state["state_hash"] = _state_hash(state)
        return state

    def restore(self, state: dict[str, Any]) -> Observation:
        self._ensure_open()
        # Accept np.load(..., allow_pickle=False) scalar values as well as JSON metadata.
        state = {
            k: (v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v)
            for k, v in state.items()
        }
        required = set(self.snapshot())
        if set(state) != required:
            raise ValueError("snapshot fields differ from schema")
        if state["state_hash"] != _state_hash(state):
            raise ValueError("snapshot content hash mismatch")
        if (
            state["schema_version"] != 1
            or state["config_json"] != self._config_json
            or state["config_hash"] != self._config_hash
            or state["mesh_hash"] != self._mesh_hash
            or state["calibration_json"] != _json(self.calibration)
        ):
            raise ValueError("snapshot scene identity mismatch")
        for key in ("faces", "edges", "rest_positions"):
            if not np.array_equal(state[key], getattr(self, key)):
                raise ValueError(f"snapshot {key} mismatch")
        positions, velocities, free = (
            np.asarray(state[key]) for key in ("positions", "velocities", "free")
        )
        if (
            positions.shape != self.rest_positions.shape
            or velocities.shape != positions.shape
            or free.shape != (len(positions),)
            or free.dtype != np.dtype(bool)
            or not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()
            or not np.isfinite(state["simulated_time"])
            or state["simulated_time"] < 0
        ):
            raise ValueError("invalid snapshot arrays or time")
        if not free.all():
            raise ValueError("cannot restore snapshot containing attached particles")
        reset_metadata = json.loads(state["reset_metadata_json"])
        if not isinstance(reset_metadata, dict):
            raise TypeError("invalid reset metadata")
        if (
            np.any(np.abs(positions[:, :2]) >= self.config.table_size_m)
            or np.any(positions[:, 2] < -0.1)
            or np.any(positions[:, 2] > 2)
        ):
            raise ValueError("snapshot outside solver domain")
        self.release()
        self.cloth.release_particle()  # Clear every flag, including reset-internal attachments.
        self.cloth.set_particles_pos(positions)
        self.cloth.set_particles_vel(velocities)
        self.simulated_time = float(state["simulated_time"])
        self.reset_metadata = reset_metadata
        self._dirty()
        if (
            not np.allclose(self.get_positions(), positions, atol=1e-7, rtol=0)
            or not np.allclose(self.get_velocities(), velocities, atol=1e-7, rtol=0)
            or not self._free().all()
        ):
            raise RuntimeError("snapshot restore readback failed")
        return self.observe()

    def reset(self, seed: int = 0, *, kind: str = "drop") -> Observation:
        self._ensure_open()
        if kind not in ("flat", "drop", "fold"):
            raise ValueError("reset kind must be flat, drop or fold")
        rng = np.random.default_rng(seed)
        self.reset_metadata = {
            "kind": kind,
            "seed": int(seed),
            "status": "preparing",
            "settle_speed_m_s": 0.01,
            "settle_hold_s": 0.5,
            "settle_budget_s": 10.0,
        }
        self.release()
        self.scene.reset()
        self.simulated_time = 0.0
        positions = self.rest_positions.copy()
        theta = float(rng.uniform(-np.pi, np.pi)) if kind != "flat" else 0.0
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        positions[:, 2] = self._initial_height
        if kind == "fold":
            # Reset-only half-fold: preserve each half's in-plane lengths, with
            # finite interlayer spacing before the bounded settling stage.
            side = positions[:, 0] > 0
            positions[side, 0] *= -1
            positions[side, 2] += self.particle_diameter * 1.5
        positions[:, :2] = positions[:, :2] @ rotation.T
        self.cloth.release_particle()
        self.cloth.set_particles_pos(positions)
        self.cloth.set_particles_vel(np.zeros_like(positions))
        self._dirty()
        try:
            if kind == "drop":
                held = int(rng.integers(len(positions)))
                target = np.array([0.0, 0.0, rng.uniform(0.5, 0.9)])
                self.reset_metadata.update(
                    {
                        "held_particle": held,
                        "suspension_target_m": target.tolist(),
                        "lift_seconds": 2.5,
                        "suspension_seconds": 0.5,
                        "lift_profile": "quintic_zero_endpoint_velocity",
                    }
                )
                self.cloth.fix_particles([held], zero_velocity=False)
                self.attached_ids = (held,)
                start = positions[held].copy()
                steps = int(np.ceil(2.5 / self.config.sim_dt))
                for i in range(steps):
                    phase = (i + 1) / steps
                    blend = phase**3 * (10 + phase * (-15 + 6 * phase))
                    endpoint = start + (target - start) * blend
                    self.set_attachment_targets(endpoint[None])
                    self.step()
                self.step(int(np.ceil(0.5 / self.config.sim_dt)))
                self.release()
            # Bounded settling, with explicit rejection instead of a flat fallback.
            stable_seconds = 0.0
            settle_steps = 0
            for settle_steps in range(1, int(np.ceil(10.0 / self.config.sim_dt)) + 1):
                self.step()
                p, v = self.get_positions(), self.get_velocities()
                if (
                    not np.isfinite(p).all()
                    or not np.isfinite(v).all()
                    or np.any(np.abs(p[:, :2]) > self.config.table_size_m / 2)
                    or np.min(p[:, 2]) < -0.01
                ):
                    raise InvalidResetError(
                        f"reset escaped table or produced invalid state: "
                        f"bounds {p.min(axis=0).tolist()} .. {p.max(axis=0).tolist()}"
                    )
                rms_speed = float(np.sqrt(np.mean(np.sum(v**2, axis=1))))
                stable_seconds = stable_seconds + self.config.sim_dt if rms_speed < 0.01 else 0.0
                if stable_seconds >= 0.5:
                    break
            self.reset_metadata.update(
                {
                    "settle_seconds": settle_steps * self.config.sim_dt,
                    "stable_seconds": stable_seconds,
                    "final_rms_speed_m_s": rms_speed,
                    "rotation_radians": theta,
                }
            )
            if stable_seconds < 0.5 or self.coverage()[1] <= 0:
                raise InvalidResetError("bounded settling budget exhausted")
            self.reset_metadata["status"] = "valid"
        except Exception as error:
            self.reset_metadata.update({"status": "invalid", "reason": str(error)})
            raise
        finally:
            self.cloth.release_particle()
            self.attached_ids = ()
            self._targets = None
            self._dirty()
        return self.observe()

    def close(self) -> None:
        if not self._closed:
            try:
                if self.scene is not None:
                    self.scene.destroy()
            finally:
                self._temporary.cleanup()
                self._closed = True
                self.attached_ids = ()
                self._targets = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
