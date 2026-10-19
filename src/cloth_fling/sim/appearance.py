"""Render-only table appearance. Importing this module never initializes Genesis."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AppearanceConfig:
    """Neutral matte laminate; texture amplitude is an absolute RGB variation."""

    color: tuple[float, float, float] = (0.55, 0.55, 0.55)
    roughness: float = 0.8
    metallic: float = 0.0
    texture_amplitude: float = 0.012
    texture_size: int = 128
    texture_seed: int = 0
    mode: str = "matte"

    def __post_init__(self):
        if self.mode not in {"matte", "legacy"}:
            raise ValueError("appearance mode must be matte or legacy")
        try:
            color = tuple(self.color)
        except TypeError as exc:
            raise ValueError("color must contain three channels") from exc
        if len(color) != 3:
            raise ValueError("color must contain three channels")
        for name, value in list(zip(("red", "green", "blue"), color)) + [
            ("roughness", self.roughness),
            ("metallic", self.metallic),
            ("texture_amplitude", self.texture_amplitude),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        if self.texture_amplitude > 0.05:
            raise ValueError("texture_amplitude must be subtle (<= 0.05)")
        if self.texture_amplitude > min(*color, *(1 - v for v in color)):
            raise ValueError("texture would clip color channels")
        for name, value, lower, upper in (
            ("texture_size", self.texture_size, 4, 4096),
            ("texture_seed", self.texture_seed, 0, 2**32 - 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        object.__setattr__(self, "color", color)

    @classmethod
    def legacy(cls):
        """Original dark table, including its unmodified raster material."""
        return cls(color=(0.04, 0.04, 0.04), texture_amplitude=0, mode="legacy")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)

    def texture_rgb(self):
        """Local seeded generator; all channels receive the same fine grain."""
        grain = np.random.default_rng(self.texture_seed).uniform(
            -self.texture_amplitude,
            self.texture_amplitude,
            (self.texture_size, self.texture_size, 1),
        )
        return np.rint((np.asarray(self.color) + grain) * 255).astype(np.uint8)


def make_surface(genesis, appearance: AppearanceConfig | None = None):
    """Before add_entity; None preserves the milestone1 legacy appearance."""
    appearance = appearance or AppearanceConfig.legacy()
    kwargs = dict(color=(*appearance.color, 1), double_sided=True)
    if appearance.mode != "legacy":
        kwargs.update(roughness=appearance.roughness, metallic=appearance.metallic)
    return genesis.surfaces.Default(**kwargs)


def install_visual_texture(scene, table, genesis, appearance: AppearanceConfig):
    """Before scene.build(): table VISUAL meshes only; collision meshes untouched.

    Use a native PBR material because Genesis 0.4.3's surface-to-SimpleMaterial
    conversion otherwise loses roughness and metallic. UVs use local table x/y.
    """
    if appearance.mode == "legacy":
        return
    if genesis.__version__ != "0.4.3" or scene.is_built:
        raise RuntimeError("texture installation requires unbuilt Genesis 0.4.3 scene")
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    if not table.vgeoms:
        raise RuntimeError("table has no visual geometry")
    for geom in table.vgeoms:
        mesh = geom.vmesh.trimesh
        xy = np.asarray(mesh.vertices)[:, :2]
        extent = np.ptp(xy, axis=0)
        if not np.isfinite(xy).all() or np.any(extent <= 0):
            raise ValueError("invalid table material coordinates")
        uv = (xy - xy.min(axis=0)) / extent
        mesh.visual = TextureVisuals(
            uv=uv,
            material=PBRMaterial(
                baseColorTexture=Image.fromarray(appearance.texture_rgb()),
                baseColorFactor=[255, 255, 255, 255],
                roughnessFactor=appearance.roughness,
                metallicFactor=appearance.metallic,
                doubleSided=True,
                alphaMode="OPAQUE",
            ),
        )


def install_native_material(scene, table, genesis, appearance: AppearanceConfig):
    """After scene.build(), before first draw: verify actual table raster material.

    This checks native factors/texture rather than trusting config.
    """
    if appearance.mode == "legacy":
        return
    if genesis.__version__ != "0.4.3" or not scene.is_built:
        raise RuntimeError("native material inspection requires built Genesis 0.4.3 scene")
    context = scene.visualizer.context
    for geom in table.vgeoms:
        node = context.rigid_nodes.get(geom.uid)
        if node is None:
            raise RuntimeError("table native raster node missing")
        for primitive in node.mesh.primitives:
            material = primitive.material
            if (
                material.roughnessFactor != appearance.roughness
                or material.metallicFactor != appearance.metallic
                or material.baseColorTexture is None
                or primitive.texcoord_0 is None
            ):
                raise RuntimeError("native table material readback failed")
