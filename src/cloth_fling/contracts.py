"""Simulator-free contracts shared by simulation, execution, and dataset tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class SceneConfig:
    cloth_size_m: float = 0.5
    mesh_spacing_m: float = 0.01
    areal_density_kg_m2: float = 0.3
    sim_dt: float = 0.002
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    after: Observation
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
    def attach(self, pixels: tuple[tuple[float, float], tuple[float, float]]) -> tuple[int, int]: ...
    def set_attachment_targets(self, targets: np.ndarray) -> None: ...
    def release(self) -> None: ...
    def step(self, count: int = 1) -> None: ...
    def coverage(self) -> tuple[float, float, float]:
        """Return (normalized coverage, projected area m2, flat reference area m2)."""
        ...
    def snapshot(self) -> dict[str, Any]: ...
    def restore(self, state: dict[str, Any]) -> Observation: ...
    def close(self) -> None: ...
