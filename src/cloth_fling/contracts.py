"""Simulator-free contracts shared by simulation, execution, and dataset tools."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class SceneConfig:
    cloth_size_m: float = 0.5
    mesh_spacing_m: float = 0.01
    areal_density_kg_m2: float = 0.3
    sim_dt: float = 0.002
    max_stretch_solver_iterations: int = 64
    table_friction: bool = True
    table_size_m: float = 1.2
    grasp_workspace_m: float = 0.9
    min_grasp_separation_m: float = 0.08
    max_grasp_separation_m: float = 0.55
    stretch_compliance: float = 1e-7
    bending_compliance: float = 1e-5
    static_friction: float = 0.15
    kinetic_friction: float = 0.15
    air_resistance: float = 0.0
    image_size: int = 256
    seed: int = 0
    # Retain the omitted-profile archive decoder. New simulation callers must
    # explicitly select native_coulomb_v2; GenesisBackend rejects both v1 laws.
    contact_profile: str = "legacy_mg_v1"
    table_static_friction: float | None = None
    table_kinetic_friction: float | None = None

    def __post_init__(self) -> None:
        positive = (
            "cloth_size_m",
            "mesh_spacing_m",
            "areal_density_kg_m2",
            "sim_dt",
            "table_size_m",
            "grasp_workspace_m",
            "min_grasp_separation_m",
            "max_grasp_separation_m",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "stretch_compliance",
            "bending_compliance",
            "static_friction",
            "kinetic_friction",
            "air_resistance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mesh_spacing_m > self.cloth_size_m / 2:
            raise ValueError("mesh spacing must resolve at least two cloth edges")
        if self.grasp_workspace_m > self.table_size_m:
            raise ValueError("grasp workspace must fit on the table")
        if self.min_grasp_separation_m >= self.max_grasp_separation_m:
            raise ValueError("minimum grasp separation must be below maximum")
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int):
            raise ValueError("image_size must be an integer")
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            isinstance(self.max_stretch_solver_iterations, bool)
            or not isinstance(self.max_stretch_solver_iterations, int)
            or not 1 <= self.max_stretch_solver_iterations <= 128
        ):
            raise ValueError("max_stretch_solver_iterations must be an integer in [1, 128]")
        if not isinstance(self.table_friction, bool):
            raise ValueError("table_friction must be a boolean")
        if self.contact_profile not in (
            "legacy_mg_v1",
            "native_coulomb_v1",
            "native_coulomb_v2",
            "reference_split_v1",
            "reference_position_coulomb_v1",
        ):
            raise ValueError("unsupported contact_profile")
        for name in ("table_static_friction", "table_kinetic_friction"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative or None")
        if (
            self.contact_profile in ("native_coulomb_v1", "native_coulomb_v2")
            or self.table_static_friction is not None
            or self.table_kinetic_friction is not None
        ) and self.table_coefficients[1] > self.table_coefficients[0]:
            raise ValueError("table kinetic friction must not exceed static friction")

    @property
    def table_coefficients(self) -> tuple[float, float]:
        """Table coefficients, separate from PBD cloth/self-contact coefficients."""
        defaults = (
            (0.40, 0.30)
            if self.contact_profile in ("native_coulomb_v1", "native_coulomb_v2")
            else (self.static_friction, self.kinetic_friction)
        )
        return tuple(
            default if value is None else value
            for default, value in zip(
                defaults, (self.table_static_friction, self.table_kinetic_friction), strict=True
            )
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Keep historical default config/snapshot identities byte-for-byte.
        if self.contact_profile == "legacy_mg_v1":
            data.pop("contact_profile")
        for name in ("table_static_friction", "table_kinetic_friction"):
            if data[name] is None:
                data.pop(name)
        return data


@dataclass
class Observation:
    observation_id: str
    rgb: np.ndarray
    depth: np.ndarray | None = None
    calibration: dict[str, Any] = field(default_factory=dict)

    def policy_rgb(self) -> np.ndarray:
        """Only this RGB array is a learned-policy observation."""
        return self.rgb.copy()


@dataclass(frozen=True)
class GraspAction:
    """Two (u, v) image pixels; top-left origin, u right, v down. World z is up."""

    pixels: tuple[tuple[float, float], tuple[float, float]]
    primitive: str = "laydown"
    policy: str = "uniform_valid"


@dataclass
class ActionOutcome:
    before: Observation
    after: Observation | None
    action: GraspAction
    coverage_before: float
    coverage_after: float
    area_before_m2: float
    area_after_m2: float
    flat_area_m2: float
    execution_valid: bool
    reason: str
    simulated_seconds: float
    wall_seconds: float
    terminated: bool = False
    truncated: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


class ClothBackend(Protocol):
    """One owner steps physics; only attached particles receive motion commands."""

    config: SceneConfig
    simulated_time: float
    attached_ids: tuple[int, ...]

    def reset(self, seed: int = 0, *, kind: str = "drop") -> Observation: ...
    def observe(self) -> Observation: ...
    def get_positions(self) -> np.ndarray: ...
    def get_velocities(self) -> np.ndarray: ...
    def attach(
        self, pixels: tuple[tuple[float, float], tuple[float, float]]
    ) -> tuple[int, int]: ...
    def set_attachment_targets(self, targets: np.ndarray) -> None: ...
    def release(self) -> None: ...
    def step(self, count: int = 1) -> None: ...
    def coverage(self) -> tuple[float, float, float]:
        """Return (normalized coverage, projected area m2, flat reference area m2)."""
        ...

    def snapshot(self) -> dict[str, Any]: ...
    def restore(self, state: dict[str, Any]) -> Observation: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ReferenceSceneConfig(SceneConfig):
    """Independent reference towel profile; no inherited square qualification."""

    profile: str = "reference_laydown_v1"
    cloth_size_m: float = 0.60
    areal_density_kg_m2: float = 0.5
    sim_dt: float = 0.001
    max_stretch_solver_iterations: int = 128
    static_friction: float = 0.6
    kinetic_friction: float = 0.6
    contact_profile: str = "reference_split_v1"
    table_static_friction: float | None = 0.3375
    table_kinetic_friction: float | None = 0.3375
    max_grasp_separation_m: float = 0.70
    cloth_width_m: float = 0.40
    cloth_mass_kg: float = 0.12
    authored_thickness_m: float = 0.002
    cloth_center_m: tuple[float, float, float] = (0.50, 0.0, 0.311)
    table_dimensions_m: tuple[float, float] = (0.80, 1.0)
    table_center_xy_m: tuple[float, float] = (0.55, 0.0)
    table_surface_z_m: float = 0.30
    table_thickness_m: float = 0.05
    coverage_bounds_m: tuple[float, float, float, float] = (0.15, 0.95, -0.40, 0.40)
    coverage_cell_m: float = 0.002
    controller_dt: float = 0.020
    substeps: int = 1
    max_bending_solver_iterations: int = 1
    stretch_relaxation: float = 0.3
    bending_relaxation: float = 0.0
    wrinkle_amplitude_m: float = 0.005
    wrinkle_seed: int = 7
    wrinkle_waves: int = 3

    def __post_init__(self):
        super().__post_init__()
        for name, count in (
            ("cloth_center_m", 3),
            ("table_dimensions_m", 2),
            ("table_center_xy_m", 2),
            ("coverage_bounds_m", 4),
        ):
            value = tuple(getattr(self, name))
            if len(value) != count or not all(math.isfinite(x) for x in value):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not all(
            math.isfinite(value)
            for value in asdict(self).values()
            if isinstance(value, (int, float))
        ):
            raise ValueError("reference numeric settings must be finite")
        if (self.profile, self.contact_profile) not in (
            ("reference_laydown_v1", "reference_split_v1"),
            ("reference_diagonal_v3", "reference_position_coulomb_v1"),
        ):
            raise ValueError("reference profile requires its split contact operator")
        for name in (
            "cloth_width_m",
            "cloth_mass_kg",
            "authored_thickness_m",
            "table_thickness_m",
            "coverage_cell_m",
            "controller_dt",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("stretch_relaxation", "bending_relaxation", "wrinkle_amplitude_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")
        for name in ("substeps", "max_bending_solver_iterations", "wrinkle_waves"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if type(self.wrinkle_seed) is not int or self.wrinkle_seed < 0:
            raise ValueError("invalid wrinkle seed")
        ticks = self.controller_dt / self.sim_dt
        if self.substeps != 1 or not math.isclose(ticks, round(ticks)) or ticks < 1:
            raise ValueError("reference controller interval must contain whole physics steps")
        if not math.isclose(
            self.cloth_mass_kg / (self.cloth_size_m * self.cloth_width_m), self.areal_density_kg_m2
        ):
            raise ValueError("reference surface density and total mass disagree")
        x0, x1, y0, y1 = self.coverage_bounds_m
        if x0 >= x1 or y0 >= y1 or min(self.table_dimensions_m) <= 0:
            raise ValueError("invalid reference bounds")

    @property
    def table_bounds_m(self):
        x, y = self.table_center_xy_m
        sx, sy = self.table_dimensions_m
        return (x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2)

    def to_dict(self):
        return asdict(self)


def scene_config_from_dict(data):
    """Decode historical square configurations or the explicit reference profile."""
    cls = (
        ReferenceSceneConfig
        if data.get("profile") in ("reference_laydown_v1", "reference_diagonal_v3")
        else SceneConfig
    )
    return cls(**data)
