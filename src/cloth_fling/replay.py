"""Re-render saved cloth trajectories without physics. ``python -m cloth_fling.replay``.

Both dataset supports and original diagnostic.json + fixture.json/npz are accepted.
Images are reconstructed privileged artifacts, not original execution capture.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cloth_fling.contracts import SceneConfig, scene_config_from_dict
from cloth_fling.data import read_rgb, read_support, read_transitions
from cloth_fling.data.storage import relative_path
from cloth_fling.rendering import VideoRecorder, file_sha256
from cloth_fling.sim.appearance import AppearanceConfig


@dataclass
class ReplayAction:
    label: str
    state: dict[str, Any]
    traces: list[dict[str, Any]]
    duration: float
    execution_valid: bool
    reason: str
    source: dict[str, Any]
    original_rgb: dict[str, str]
    table_height_m: float = 0.0


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def validate_action(action):
    """Fail on malformed trajectories before allocating a simulator or video writer."""
    action.duration = _finite_number(action.duration, "duration")
    positions = np.asarray(action.state["positions"])
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("snapshot positions must be N x 3")
    if not np.isfinite(positions).all():
        raise ValueError("snapshot positions must be finite")
    scene_config_from_dict(json.loads(str(action.state["config_json"])))
    previous = -1.0
    if not action.traces:
        # Static terminal geometry is explicit, never fabricated action motion.
        action.traces = [
            {
                "phase": "terminal_only_no_trajectory",
                "simulated_seconds": 0.0,
                "cloth_positions_m": positions.copy(),
                "attachment_ids": [],
                "coverage": None,
            }
        ]
    for trace in action.traces:
        sec = _finite_number(trace["simulated_seconds"], "trace time")
        if sec < previous or sec > action.duration + 1e-6:
            raise ValueError("trace times must be ordered and within action duration")
        previous = sec
        p = np.asarray(trace["cloth_positions_m"])
        if p.shape != positions.shape or not np.isfinite(p).all():
            raise ValueError("trace geometry does not match snapshot")
        if "cloth_velocities_m_s" in trace:
            v = np.asarray(trace["cloth_velocities_m_s"])
            if v.shape != p.shape or not np.isfinite(v).all():
                raise ValueError("invalid trace velocities")
        from cloth_fling.rendering import marker_positions

        marker_positions(p, trace.get("attachment_ids", []), str(trace["phase"]))
    return action


def _diagnostic_fixture(directory, references):
    metadata = relative_path(directory, references.get("metadata", "fixture.json"))
    arrays = relative_path(directory, references.get("arrays", "fixture.npz"))
    document = json.loads(metadata.read_text())
    with np.load(arrays, allow_pickle=False) as archive:

        def decode(value):
            if isinstance(value, dict):
                if set(value) == {"npz_key"}:
                    return archive[value["npz_key"]].copy()
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value

        return decode(document)


def load_replay(source, *, episode_id=None):
    """Offline loader. Retain invalid attempts and group dataset rows by episode/step."""
    source = Path(source).resolve()
    actions = []
    if source.is_dir() and (source / "contact-report.json").is_file():
        from cloth_fling.reference_diagonal_replay import load_diagonal_replay

        if episode_id is not None:
            raise ValueError("episode_id selection applies to datasets only")
        actions, origin = load_diagonal_replay(source)
    elif source.is_dir() and (source / "tune-report.json").is_file():
        from cloth_fling.reference_tune import read_hashed_bundle

        if episode_id is not None:
            raise ValueError("episode_id selection applies to datasets only")
        document = json.loads((source / "tune-report.json").read_text())
        state = read_hashed_bundle(source, "fixture.json")
        for attempt in document["attempts"]:
            ref = attempt["outcome"]
            if file_sha256(relative_path(source, ref["path"])) != ref["sha256"]:
                raise ValueError("tuning outcome hash mismatch")
            run = read_hashed_bundle(source, ref["path"])
            actions.append(
                validate_action(
                    ReplayAction(
                        label=f"{attempt['category']} {attempt['attempt']}/c{attempt['candidate']} "
                        + ("PASS" if attempt.get("passed") else "FAIL"),
                        state=state,
                        traces=run["evidence"]["phase_traces"],
                        duration=run["simulated_seconds"],
                        execution_valid=run["execution_valid"],
                        reason=run["reason"],
                        source=run,
                        original_rgb={},
                        table_height_m=0.30,
                    )
                )
            )
        origin = {
            "format": "reference_tuning",
            "source_tune_report_sha256": file_sha256(source / "tune-report.json"),
        }
    elif source.is_dir() and (source / "reference-report.json").is_file():
        from cloth_fling.data.storage import read_bundle

        if episode_id is not None:
            raise ValueError("episode_id selection applies to datasets only")
        document = json.loads((source / "reference-report.json").read_text())

        def checked_bundle(reference):
            path = relative_path(source, reference["path"])
            if file_sha256(path) != reference["sha256"]:
                raise ValueError("reference bundle hash mismatch")
            bundle = json.loads(path.read_text())
            arrays = bundle.get("arrays")
            if arrays and file_sha256(relative_path(source, arrays["path"])) != arrays["sha256"]:
                raise ValueError("reference arrays hash mismatch")
            return read_bundle(source, reference["path"])

        run = checked_bundle(document["single_action"])
        state = checked_bundle(document["initial_state"])
        evidence = run["evidence"]
        actions.append(
            validate_action(
                ReplayAction(
                    label="reference controlled laydown",
                    state=state,
                    traces=evidence.get("phase_traces", []),
                    duration=run["simulated_seconds"],
                    execution_valid=run["execution_valid"],
                    reason=run["reason"],
                    source=run,
                    original_rgb={},
                    table_height_m=0.30,
                )
            )
        )
        origin = {
            "format": "reference_diagnostic",
            "source_reference_report_sha256": file_sha256(source / "reference-report.json"),
        }
    elif source.is_dir() and (source / "manifest.json").is_file():
        manifest = json.loads((source / "manifest.json").read_text())
        episodes = {e["episode_id"]: e for e in manifest["episodes"]}
        rows = sorted(
            read_transitions(source).to_pylist(),
            key=lambda row: (row["episode_id"], row["step_id"]),
        )
        for row in rows:
            if episode_id is not None and row["episode_id"] != episode_id:
                continue
            final = row.get("final_state_path")
            if final:
                state = read_support(source, final)
            else:
                reset = read_support(source, episodes[row["episode_id"]]["reset"]["path"])
                state = reset.get("state", reset)
            evidence = read_support(source, row["evidence_path"])["evidence"]
            traces = evidence.get("phase_traces", [])
            action = ReplayAction(
                label=f"episode {row['episode_id']} step {row['step_id']}",
                state=state,
                traces=traces,
                duration=row.get("simulated_seconds") or 0.0,
                execution_valid=row["execution_valid"],
                reason=row["reason"],
                source=row,
                original_rgb={
                    name: row[f"{name}_rgb_path"]
                    for name in ("before", "after")
                    if row.get(f"{name}_rgb_path")
                },
                table_height_m=evidence.get("config", {}).get("table_height_m", 0.0),
            )
            actions.append(validate_action(action))
        origin = {
            "format": "dataset",
            "source_manifest_sha256": file_sha256(source / "manifest.json"),
        }
    else:
        if episode_id is not None:
            raise ValueError("episode_id selection applies to datasets only")
        source = source / "diagnostic.json" if source.is_dir() else source
        document = json.loads(source.read_text())
        if document.get("schema") != "cloth_fling.diagnostic.v1":
            raise ValueError("expected dataset root or cloth_fling.diagnostic.v1 JSON")
        state = _diagnostic_fixture(source.parent, document.get("fixture_artifacts", {}))
        for index, run in enumerate(document["runs"]):
            evidence = run["evidence"]
            artifacts = run.get("artifacts", {})
            actions.append(
                validate_action(
                    ReplayAction(
                        label=f"diagnostic {index} {run['primitive']}",
                        state=state,
                        traces=evidence.get("phase_traces", []),
                        duration=run["simulated_seconds"],
                        execution_valid=run["execution_valid"],
                        reason=run["reason"],
                        source=run,
                        original_rgb={
                            name: artifacts[f"{name}_rgb"]
                            for name in ("before", "after")
                            if artifacts.get(f"{name}_rgb")
                        },
                        table_height_m=evidence.get("config", {}).get("table_height_m", 0.0),
                    )
                )
            )
        origin = {
            "format": "diagnostic",
            "source_diagnostic_sha256": file_sha256(source),
            "fixture_hashes": {
                name: file_sha256(relative_path(source.parent, ref))
                for name, ref in document.get(
                    "fixture_artifacts", {"metadata": "fixture.json", "arrays": "fixture.npz"}
                ).items()
            },
        }
    if not actions:
        raise ValueError("no replay actions match the requested source/episode")
    origin.update(
        actions=len(actions),
        invalid_attempts=sum(not action.execution_valid for action in actions),
        source_engines=list(
            {
                json.dumps(
                    source_engine_identity(action.state), sort_keys=True
                ): source_engine_identity(action.state)
                for action in actions
            }.values()
        ),
        runtime_qualification_inherited=False,
    )
    return actions, origin


def sample_traces(action, fps):
    """Zero-order hold; never interpolate or synthesize cloth dynamics."""
    fps = _finite_number(fps, "fps")
    if not 0 < fps <= 240:
        raise ValueError("fps must be in (0, 240]")
    times = np.array([trace["simulated_seconds"] for trace in action.traces])
    if times[0] > 1e-6:
        raise ValueError("trajectory has no initial sample at t=0")
    # Include terminal geometry once, even when duration is off the video clock.
    count = int(math.ceil(action.duration * fps))
    for index in range(count + 1):
        sec = min(index / fps, action.duration)
        sample = copy.copy(
            action.traces[max(0, int(np.searchsorted(times, sec, side="right")) - 1)]
        )
        sample["source_trace_seconds"] = sample["simulated_seconds"]
        sample["simulated_seconds"] = sec
        yield sample


def _forbidden_step(*_, **__):
    raise RuntimeError("reconstructed replay must never step physics")


def source_engine_identity(state):
    """Historical schema 1 is evidence of the old engine, never a new restore state."""
    schema = int(state.get("schema_version", 1))
    if schema == 1:
        return {
            "name": "genesis",
            "version": "0.4.3",
            "runtime_generation": "genesis-0.4.3",
            "snapshot_schema_version": 1,
        }
    if schema == 2 and "runtime_identity_json" in state:
        identity = json.loads(str(state["runtime_identity_json"]))
        if (
            not isinstance(identity, dict)
            or identity.get("engine") != "genesis"
            or not identity.get("engine_version")
        ):
            raise ValueError("invalid source runtime identity")
        return {
            "name": "genesis",
            "version": identity["engine_version"],
            "runtime_generation": f"genesis-{identity['engine_version']}",
            "snapshot_schema_version": 2,
            "backend_identity": identity,
        }
    raise ValueError("snapshot is missing its source engine identity")


def render_recorded_state(env, state, trace):
    """Explicit render-only import; never rewrite a legacy snapshot into a new schema.

    The backend hook owns device writes and independent topology verification.
    Validate here too, before allowing even that hook to touch the scene.
    """
    for key in ("faces", "rest_positions"):
        if key not in state:
            raise ValueError(f"render-only replay requires recorded {key}")
        if not np.array_equal(np.asarray(state[key]), getattr(env, key)):
            raise ValueError(f"render-only replay {key} mismatch; remapping is forbidden")
    positions = np.asarray(trace["cloth_positions_m"])
    if (
        positions.shape != np.asarray(state["rest_positions"]).shape
        or not np.isfinite(positions).all()
    ):
        raise ValueError("render-only replay positions do not match recorded mesh")
    hook = getattr(env, "restore_recorded_geometry", None)
    if not callable(hook):
        raise RuntimeError(
            "backend lacks restore_recorded_geometry render-only hook; physical restore is forbidden"
        )
    # Import the untouched original envelope. The hook validates its content hash
    # and exact mesh, and latches the backend into render-only mode. Keep that
    # latch across frames; update geometry through the native public setter.
    if getattr(env, "_replay_source_state", None) is not state:
        hook(state)
        env._replay_source_state = state
    env.cloth.set_particles_pos(positions.copy())
    if not np.array_equal(env.get_positions(), positions):
        raise RuntimeError("render-only geometry readback differs from recorded positions")


def render_replay(
    source,
    output_dir,
    *,
    fps=20,
    episode_id=None,
    appearance=None,
    max_frames=10000,
    backend_factory=None,
    writer_factory=None,
):
    """Return hashed provenance; never open a browser, step, reset, or mutate source."""
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    actions, origin = load_replay(source, episode_id=episode_id)
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    source_root = source if source.is_dir() else source.parent
    if output_dir.is_relative_to(source_root):
        raise ValueError("write replay outside the source dataset/diagnostic directory")
    if (output_dir / "replay-provenance.json").exists() or (
        output_dir / "video-metadata.json"
    ).exists():
        raise FileExistsError("replay output already exists; choose a fresh directory")
    native_template = backend_factory is None
    if native_template:
        from cloth_fling.sim.backend import GenesisBackend

        backend_factory = GenesisBackend
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    truncated = False
    with VideoRecorder(
        output_dir,
        fps=fps,
        appearance=appearance,
        kind="reconstructed_replay",
        writer_factory=writer_factory,
    ) as recorder:
        reference_scene = any(
            json.loads(str(action.state["config_json"])).get("profile")
            in ("reference_laydown_v1", "reference_diagonal_v3")
            for action in actions
        )
        if origin["format"] in ("reference_tuning", "reference_diagonal") or reference_scene:
            recorder.view_poses = {
                "overhead": {"pos": (0.55, 0, 2.9), "lookat": (0.55, 0, 0.4), "up": (0, 1, 0)},
                "oblique": {"pos": (1.9, -1.9, 1.7), "lookat": (0.50, 0, 0.5), "up": (0, 0, 1)},
            }
        for index, action in enumerate(actions):
            template = json.loads(str(action.state["config_json"]))
            reference = template.get("profile") in ("reference_laydown_v1", "reference_diagonal_v3")
            if native_template and not reference:
                # The new backend rejects archival contact laws. This selects
                # its current template only; replay never executes that law.
                template["contact_profile"] = "native_coulomb_v2"
            config = scene_config_from_dict(template) if reference else SceneConfig(**template)
            factory = backend_factory
            if native_template and reference:
                from cloth_fling.sim.reference import ReferenceBackend

                factory = ReferenceBackend
            kwargs = {} if appearance is None else {"appearance": appearance}
            with factory(config, **kwargs) as env:
                # Defense in depth: refuse scene AND wrapper stepping/reset paths.
                env.step = _forbidden_step
                env.reset = _forbidden_step
                env.scene.step = _forbidden_step
                env.scene.reset = _forbidden_step
                env.restore = _forbidden_step
                source_engine = source_engine_identity(action.state)
                recorder.source_engines[json.dumps(source_engine, sort_keys=True)] = source_engine
                recorder.source_configs[action.label] = json.loads(str(action.state["config_json"]))
                recorder.start_action(action.label)
                recorder.table_height_m = action.table_height_m
                recorder.particle_radius_m = (
                    getattr(env, "particle_diameter", config.mesh_spacing_m) / 2
                )
                for trace in sample_traces(action, fps):
                    if len(recorder.samples) >= max_frames:
                        truncated = True
                        break
                    render_recorded_state(env, action.state, trace)
                    recorder.capture(env, trace, replay=True)
            for name, reference in action.original_rgb.items():
                from PIL import Image

                target = output_dir / f"action-{index}-{name}-original.png"
                if origin["format"] == "dataset":
                    rgb = read_rgb(source_root, reference)
                    Image.fromarray(rgb).save(target)
                else:
                    original = relative_path(source_root, reference)
                    with Image.open(original) as image:
                        image.convert("RGB").save(target)
                copied.append(
                    {
                        "path": target.name,
                        "sha256": file_sha256(target),
                        "kind": "original_observation",
                        "source": reference,
                    }
                )
            if truncated:
                break
    metadata = recorder.metadata()
    metadata.update(origin)
    metadata.update(
        original_video_recorded=False,
        physics_reexecuted=False,
        geometry_import="render-only; exact recorded faces and rest vertices; no physical restore",
        trajectory_sampling="last saved mesh at or before frame time; no interpolation",
        original_observations=copied,
        truncated=truncated,
        max_frames=max_frames,
        view_poses=getattr(recorder, "view_poses", None),
        action_outcomes=[
            {"action": a.label, "execution_valid": a.execution_valid, "reason": a.reason}
            for a in actions
        ],
    )
    (output_dir / "replay-provenance.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="dataset root or diagnostic JSON/directory")
    parser.add_argument("--output", type=Path, help="fresh directory outside source")
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--appearance", help="legacy (default), matte, or appearance JSON")
    parser.add_argument(
        "--inspect", action="store_true", help="validate trajectories without Genesis"
    )
    args = parser.parse_args(argv)
    try:
        if args.inspect:
            actions, metadata = load_replay(args.source, episode_id=args.episode_id)
            metadata["durations_seconds"] = [a.duration for a in actions]
        else:
            if args.output is None:
                parser.error("--output is required when rendering")
            appearance = None
            if args.appearance == "matte":
                appearance = AppearanceConfig()
            elif args.appearance not in (None, "legacy"):
                appearance = AppearanceConfig.from_json(args.appearance)
            metadata = render_replay(
                args.source,
                args.output,
                fps=args.fps,
                episode_id=args.episode_id,
                appearance=appearance,
                max_frames=args.max_frames,
            )
        print(json.dumps(metadata, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(f"replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
