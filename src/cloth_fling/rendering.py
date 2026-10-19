"""Native debug capture and optional video writing; no simulator imports at import time.

These images are privileged evaluation artifacts, never policy observations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MARKER_RADIUS_M = 0.01
MARKER_COLORS = ((1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0))
OBLIQUE_POSE = {"pos": (0.95, -1.15, 0.85), "lookat": (0, 0, 0.13), "up": (0, 0, 1)}


def _points(value):
    points = np.asarray(value, dtype=float)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("debug markers require two finite world XYZ positions")
    return points.copy()


def marker_positions(positions, attachment_ids, phase, selected_targets_m=None):
    """Measured vertices when attached; selected surface targets only before attachment.

    Empty attachments in release/settle/terminal phases always hide markers, even
    when historical measured/commanded positions remain in a trace.
    """
    ids = tuple(attachment_ids)
    if ids:
        if (
            len(ids) != 2
            or len(set(ids)) != 2
            or any(
                isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in ids
            )
        ):
            raise ValueError("debug attachment IDs must be two distinct integers")
        positions = np.asarray(positions)
        if min(ids) < 0 or max(ids) >= len(positions):
            raise ValueError("debug attachment ID outside mesh")
        return _points(positions[list(ids)])
    if phase in {"before", "selected", "pre_attach"} and selected_targets_m is not None:
        return _points(selected_targets_m)
    return None


@contextmanager
def debug_markers(scene, positions):
    """Scope native debug objects only; preserve unrelated debug objects on failure."""
    with ExitStack() as cleanup:
        if positions is not None:
            for point, color in zip(_points(positions), MARKER_COLORS, strict=True):
                obj = scene.draw_debug_sphere(pos=point, radius=MARKER_RADIUS_M, color=color)
                cleanup.callback(scene.clear_debug_object, obj)
        yield


@contextmanager
def preserved_camera(camera):
    """Restore public transform and Genesis 0.4.3 redundant pose fields exactly."""
    if getattr(camera, "_in_recording", False):
        raise RuntimeError("debug capture requires camera native recording to be disabled")
    transform = np.asarray(camera.transform).copy()
    fields = {}
    for name in ("_pos", "_lookat", "_up", "_transform", "_quat"):
        if hasattr(camera, name):
            value = getattr(camera, name)
            fields[name] = value.clone() if hasattr(value, "clone") else copy.deepcopy(value)
    try:
        yield
    finally:
        camera.set_pose(transform=transform)
        for name, value in fields.items():
            getattr(camera, name)[...] = value


def render_debug_views(env, *, phase, selected_targets_m=None, attachment_ids=None):
    """Capture overhead + oblique without observe(), stepping, or cache mutation.

    Run synchronously on the scene owner thread. ``attachment_ids`` overrides are
    reserved for reconstructed replay of saved attachment IDs; live recording
    always uses env.attached_ids. No debug depth is requested or returned.
    """
    ids = env.attached_ids if attachment_ids is None else attachment_ids
    points = marker_positions(env.get_positions(), ids, phase, selected_targets_m)
    options = dict(rgb=True, depth=False, antialiasing=False, force_render=True)
    with preserved_camera(env.camera):
        with debug_markers(env.scene, points):
            # First offscreen call may only allocate native dynamic vertex buffers.
            env.camera.render(**options)
            overhead = np.asarray(env.camera.render(**options)[0], dtype=np.uint8).copy()
            env.camera.set_pose(**OBLIQUE_POSE)
            oblique = np.asarray(env.camera.render(**options)[0], dtype=np.uint8).copy()
    for frame in (overhead, oblique):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("native render must return H x W x 3 RGB")
    return {"overhead": overhead, "oblique": oblique}


def annotate_frame(rgb, *, label, action, phase, coverage=None, min_clearance_m=None):
    """Copy pixels before drawing; caller-owned RGB buffers never receive overlays."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    coverage_text = "unknown" if coverage is None else f"{coverage * 100:.1f}%"
    clearance_text = "unknown" if min_clearance_m is None else f"{min_clearance_m * 1000:.1f} mm"
    lines = [
        label,
        f"action {action} | {phase}",
        f"coverage {coverage_text}",
        f"min cloth clearance {clearance_text}",
    ]
    draw.rectangle((0, 0, image.width, 60), fill=(16, 20, 28))
    for index, line in enumerate(lines):
        draw.text((5, 2 + 14 * index), line, fill="white")
    return np.asarray(image).copy()


class FrameWriter:
    """Lazy imageio/FFmpeg MP4 writer with fixed RGB shape and reliable close."""

    def __init__(self, path, *, fps=20.0, writer_factory=None):
        if isinstance(fps, bool) or not math.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError("fps must be finite in (0, 240]")
        self.path, self.fps = Path(path), float(fps)
        self.frame_count, self.shape = 0, None
        self._writer, self._factory, self._closed = None, writer_factory, False

    def append(self, frame):
        if self._closed:
            raise RuntimeError("frame writer is closed")
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("video frame must be uint8 H x W x 3 RGB")
        if min(frame.shape[:2]) < 2 or any(v % 2 for v in frame.shape[:2]):
            raise ValueError("H264 video requires positive even image dimensions")
        if self.shape is not None and frame.shape != self.shape:
            raise ValueError("video frame shape changed")
        if self._writer is None:
            factory = self._factory
            if factory is None:
                try:
                    import imageio.v2 as imageio
                    import imageio_ffmpeg  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError("Install cloth-fling[video] to write MP4 files") from exc
                factory = imageio.get_writer
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = factory(
                self.path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                ffmpeg_log_level="error",
                pixelformat="yuv420p",
            )
            self.shape = frame.shape
        self._writer.append_data(frame.copy())
        self.frame_count += 1

    def close(self):
        if not self._closed:
            self._closed = True
            if self._writer is not None:
                self._writer.close()

    def metadata(self):
        return {
            "path": self.path.name,
            "fps": self.fps,
            "frames": self.frame_count,
            "dimensions": list(self.shape[:2][::-1]) if self.shape else None,
            "sha256": file_sha256(self.path) if self.path.is_file() else None,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoRecorder:
    """Callable sampled-trace observer for execute_action(trace_callback=recorder).

    Capture occurs DURING execution; fps is a maximum capture rate, not a promise
    of missing intermediate states. Actual frame times are preserved in metadata.
    Callback and encoder failures propagate to the execution owner.
    """

    def __init__(
        self,
        output_dir,
        *,
        fps=20,
        appearance=None,
        kind="native_render_capture",
        writer_factory=None,
        table_height_m=0.0,
        particle_radius_m=0.0,
    ):
        if kind not in {"native_render_capture", "reconstructed_replay"}:
            raise ValueError("unknown video evidence kind")
        self.output_dir = Path(output_dir)
        self.kind, self.appearance = kind, appearance
        for value in (table_height_m, particle_radius_m):
            if not math.isfinite(value):
                raise ValueError("clearance geometry must be finite")
        if particle_radius_m < 0:
            raise ValueError("particle radius must be nonnegative")
        self.table_height_m, self.particle_radius_m = table_height_m, particle_radius_m
        self.writers = {
            name: FrameWriter(
                self.output_dir / f"{name}.mp4", fps=fps, writer_factory=writer_factory
            )
            for name in ("overhead", "oblique", "combined")
        }
        self.fps = self.writers["combined"].fps
        self.samples, self.action, self._last_time = [], 0, None
        self.physics_configs = {}
        self._closed, self._complete = False, False
        self._failure = None

    def start_action(self, action, *, selected_targets_m=None):
        self.action = action
        self.selected_targets_m = (
            None if selected_targets_m is None else _points(selected_targets_m)
        )
        self._last_time = None

    def __call__(self, env, trace):
        try:
            return self.capture(env, trace)
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            raise

    def capture(self, env, trace, *, replay=False):
        if self._closed:
            raise RuntimeError("video recorder is closed")
        sec = float(trace["simulated_seconds"])
        if not math.isfinite(sec) or sec < 0:
            raise ValueError("trace time must be finite and nonnegative")
        phase = str(trace["phase"])
        if self._last_time is not None and not replay:
            if sec < self._last_time:
                raise ValueError("trace time reversed; call start_action for each action")
            if sec - self._last_time < 1 / self.fps - 1e-9 and (
                self.samples and self.samples[-1]["phase"] == phase
            ):
                return
        selected = trace.get("selected_grasp_positions_m")
        if selected is None:
            selected = getattr(self, "selected_targets_m", None)
        ids = trace.get("attachment_ids", ()) if replay else None
        views = render_debug_views(
            env, phase=phase, selected_targets_m=selected, attachment_ids=ids
        )
        positions = env.get_positions()
        clearance = float(np.min(positions[:, 2]) - self.table_height_m - self.particle_radius_m)
        coverage = trace.get("coverage")
        if coverage is not None and not math.isfinite(float(coverage)):
            coverage = None
        frames = {}
        for name, rgb in views.items():
            rgb = np.asarray(Image.fromarray(rgb).resize((384, 384), Image.Resampling.LANCZOS))
            frames[name] = annotate_frame(
                rgb,
                label=f"{self.kind} | {name}",
                action=self.action,
                phase=phase,
                coverage=coverage,
                min_clearance_m=clearance,
            )
            self.writers[name].append(frames[name])
        combined = np.concatenate((frames["overhead"], frames["oblique"]), axis=1)
        self.writers["combined"].append(combined)
        if not self.samples:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(combined).save(self.output_dir / "preview.png")
        config_json = json.dumps(
            env.config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        self.physics_configs[config_hash] = env.config.to_dict()
        self.samples.append(
            {
                "physics_config_sha256": config_hash,
                "action": self.action,
                "phase": phase,
                "trace_seconds": sec,
                "coverage": coverage,
                "min_cloth_clearance_m": clearance,
                "world_min_vertex_z_m": float(np.min(positions[:, 2])),
            }
        )
        self._last_time = sec

    def close(self, *, complete=True):
        if self._closed:
            return self.metadata()
        self._closed = True
        try:
            with ExitStack() as cleanup:
                for writer in self.writers.values():
                    cleanup.callback(writer.close)
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._complete = bool(complete and self._failure is None)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            metadata = self.metadata()
            (self.output_dir / "video-metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
        return metadata

    def metadata(self):
        return {
            "kind": self.kind,
            "complete": self._complete,
            "failure": self._failure,
            "physics_reexecuted": False if self.kind == "reconstructed_replay" else None,
            "captured_during_execution": self.kind == "native_render_capture",
            "appearance": (
                self.appearance.to_dict() if self.appearance is not None else {"mode": "legacy"}
            ),
            "physics_configs": self.physics_configs,
            "markers": {"radius_m": MARKER_RADIUS_M, "positions": "measured when attached"},
            "sampling": "sampled traces at nominal encoded fps; see actual trace timestamps",
            "table_height_m": self.table_height_m,
            "particle_radius_m": self.particle_radius_m,
            "videos": [
                {
                    **w.metadata(),
                    "kind": self.kind,
                    "captured_during_execution": self.kind == "native_render_capture",
                    "physics_config_sha256": list(self.physics_configs),
                    "appearance": self.appearance.to_dict()
                    if self.appearance
                    else {"mode": "legacy"},
                }
                for w in self.writers.values()
            ],
            "samples": self.samples,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.close(complete=exc_type is None)
