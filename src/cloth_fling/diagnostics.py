"""Bounded restored-fixture diagnostics, never an automatic quality qualification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from cloth_fling.contracts import ClothBackend, GraspAction
from cloth_fling.provenance import config_provenance, execute_recorded, runtime_provenance
from cloth_fling.selection import feasible_candidates, pixels_to_world, select_action
from cloth_fling.sim.mesh import pixel_ray, ray_surface_hit, resolve_grasps
from cloth_fling.sim.primitives import PrimitiveConfig, execute_action


def reviewed_pair_candidates(
    env: ClothBackend,
    *,
    max_pairs: int = 8,
    candidate_budget: int = 256,
) -> list[dict[str, Any]]:
    """Privileged diagnostic ranking for human RGB review; never execute a pair.

    Rerank the bounded widest conservative RGB boundary candidates by material
    rest chord. Resolve with the *same* first-hit, visible-vertex snapping and
    feasibility checks as the backend, not nearest hidden particles. This is not
    a learned policy, a global search, or a guarantee of unfolding behavior.
    """
    if any(type(v) is not int or v < 1 for v in (max_pairs, candidate_budget)):
        raise ValueError("max_pairs and candidate_budget must be positive integers")
    if tuple(env.attached_ids):
        raise ValueError("reviewed_pair_requires_unattached_fixture")
    observation = env.observe()
    positions = np.asarray(env.get_positions(), dtype=float)
    rest, faces = getattr(env, "rest_positions", None), getattr(env, "faces", None)
    if rest is None or faces is None:
        metadata = env.snapshot()
        rest, faces = metadata.get("rest_positions", rest), metadata.get("faces", faces)
    if rest is None or faces is None:
        raise ValueError("reviewed_pair_requires_rest_positions_and_faces")
    rest, faces = np.asarray(rest, dtype=float), np.asarray(faces)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or len(positions) < 2
        or rest.shape != positions.shape
        or not np.isfinite([positions, rest]).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
        or not len(faces)
        or np.any(faces < 0)
        or np.any(faces >= len(positions))
    ):
        raise ValueError("invalid_reviewed_pair_geometry")
    config = env.config
    candidates = feasible_candidates(observation, config, max_pairs=candidate_budget)
    rows = []
    for action in candidates:
        try:
            ids = resolve_grasps(
                action.pixels,
                positions,
                faces,
                observation.calibration,
                radius=config.mesh_spacing_m * 1.5,
                workspace=config.grasp_workspace_m,
                min_separation=config.min_grasp_separation_m,
                max_separation=config.max_grasp_separation_m,
            )
        except ValueError:
            # A rendered boundary can disagree with a ray hit. Never replace it
            # with a hidden particle or relax the canonical resolver's bounds.
            continue
        hits = np.array(
            [
                ray_surface_hit(*pixel_ray(pixel, observation.calibration), positions, faces)[0]
                for pixel in action.pixels
            ]
        )
        surface = pixels_to_world(observation, np.asarray(action.pixels))
        rows.append(
            {
                "pixels": [list(pixel) for pixel in action.pixels],
                "predicted_grasp_particle_ids": list(ids),
                "snap_distances_m": np.linalg.norm(positions[list(ids)] - hits, axis=1).tolist(),
                "visible_span_m": float(np.linalg.norm(surface[1] - surface[0])),
                "resolved_span_m": float(np.linalg.norm(positions[ids[1]] - positions[ids[0]])),
                "rest_flat_chord_m": float(np.linalg.norm(rest[ids[1]] - rest[ids[0]])),
                "observation_id": observation.observation_id,
                "policy": "reviewed_pair_privileged_rest_geometry",
                "requires_visual_review": True,
                "candidate_budget": candidate_budget,
                "candidate_pairs_considered": len(candidates),
            }
        )
    rows.sort(key=lambda r: (-r["rest_flat_chord_m"], -r["visible_span_m"], r["pixels"]))
    selected, seen = [], set()
    for row in rows:
        key = tuple(sorted(row["predicted_grasp_particle_ids"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == max_pairs:
            break
    return selected


def _fingerprint(env):
    arrays = [env.get_positions(), env.get_velocities()]
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite_fixture_state")
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    digest.update(
        repr((float(env.simulated_time), tuple(env.attached_ids), tuple(env.coverage()))).encode()
    )
    return digest.hexdigest()


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _save_fixture(directory, fixture):
    arrays = {}

    def split(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("fixture object arrays are not safe state artifacts")
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"npz_key": key}
        if isinstance(value, dict):
            return {str(k): split(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [split(v) for v in value]
        return _json_safe(value)

    metadata = split(fixture)
    np.savez_compressed(directory / "fixture.npz", **arrays)
    (directory / "fixture.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {"metadata": "fixture.json", "arrays": "fixture.npz"}


def _save_observations(directory, primitive, outcome):
    from PIL import Image

    references = {}
    depths = {}
    cameras = {}
    for name, obs in (("before", outcome.before), ("after", outcome.after)):
        if obs is None:
            references[f"{name}_rgb"] = None
            cameras[name] = None
            continue
        filename = f"{primitive}_{name}.png"
        rgb = np.asarray(obs.rgb)
        if np.issubdtype(rgb.dtype, np.floating) and rgb.max(initial=0) <= 1:
            rgb = rgb * 255
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(directory / filename)
        references[f"{name}_rgb"] = filename
        if obs.depth is not None:
            depths[name] = obs.depth
        cameras[name] = _json_safe(obs.calibration)
    depth_file, camera_file = f"{primitive}_depth.npz", f"{primitive}_camera.json"
    np.savez_compressed(directory / depth_file, **depths)
    (directory / camera_file).write_text(json.dumps(cameras, indent=2, allow_nan=False) + "\n")
    references.update(depth=depth_file, calibration=camera_file)
    return references


def run_diagnostic(
    env: ClothBackend,
    *,
    seed: int = 0,
    output_dir=None,
    primitive_config: PrimitiveConfig | None = None,
    action: GraspAction | None = None,
    primitives: tuple[str, ...] = ("laydown", "fling"),
    record_video=False,
    appearance=None,
    recorder_factory=None,
) -> dict[str, Any]:
    """Run the requested primitives from the *current* saved fixture.

    No reset or broad campaign. Select once before any motion. Restore and verify positions,
    velocities, attachments, clock and coverage exactly before each action and on exit.
    A restoration mismatch stops comparison and is explicitly invalid, never a new fixture.
    """
    if isinstance(primitives, str):
        raise ValueError("primitives must be a nonempty sequence of distinct laydown/fling names")
    try:
        primitives = tuple(primitives)
    except TypeError as exc:
        raise ValueError("primitives must be a sequence of laydown/fling names") from exc
    if (
        not primitives
        or any(not isinstance(p, str) or p not in {"laydown", "fling"} for p in primitives)
        or len(set(primitives)) != len(primitives)
    ):
        raise ValueError("primitives must be a nonempty sequence of distinct laydown/fling names")
    if record_video and output_dir is None:
        raise ValueError("record_video requires output_dir")
    if record_video:
        for primitive in primitives:
            if (Path(output_dir) / "video" / f"diagnostic-{primitive}").exists():
                raise FileExistsError("diagnostic video output exists; choose a fresh directory")
    started = time.monotonic()
    c = primitive_config or PrimitiveConfig()
    directory = Path(output_dir) if output_dir is not None else None
    report: dict[str, Any] = {
        "schema": "cloth_fling.diagnostic.v1",
        "runtime_provenance": runtime_provenance(),
        "provenance": config_provenance(env, appearance=appearance, primitive_config=c),
        "recordings": [],
        "seed": seed,
        "primitives": list(primitives),
        "config": asdict(c),
        "runs": [],
        "errors": [],
        "execution_valid": False,
        "restore_checks": [],
        "final_restore_valid": False,
        "restore_validation_fields": [
            "positions",
            "velocities",
            "attachments",
            "simulated_time",
            "coverage",
        ],
        "pair_selection": "geometry_guided_privileged_depth; RGB boundary candidates",
        "comparison": (
            "one laydown baseline / one retained-grasp fling candidate"
            if len(primitives) == 2
            else f"one {primitives[0]} execution; no paired comparison"
        ),
        "quality_qualified": False,
        "behavior_passed": None,
        "evidence_scope": "backend execution diagnostic; no GPU or behavior qualification implied",
        "campaign_deferred": "600-run campaign, training, held-out policy comparisons",
    }
    fixture = None
    fingerprint = None
    try:
        if tuple(env.attached_ids):
            raise ValueError("diagnostic_requires_unattached_fixture")
        fixture = copy.deepcopy(env.snapshot())
        fingerprint = _fingerprint(env)
        report["provenance"] = config_provenance(
            env,
            appearance=appearance,
            primitive_config=c,
            reset_source=fixture,
        )
        if action is None:
            action = select_action(
                env.observe(),
                env.config,
                policy="geometry_guided",
                primitive=primitives[0],
                rng=np.random.default_rng(seed),
            )
        else:
            report["pair_selection"] = "manual_fixed_pair; privileged depth execution feasibility"
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            report["fixture_artifacts"] = _save_fixture(directory, fixture)
        report["fixed_pixels"] = action.pixels
        report["fixture_sha256"] = fingerprint
        for primitive in primitives:
            if time.monotonic() - started >= len(primitives) * c.max_wall_seconds:
                raise TimeoutError("diagnostic_wall_deadline")
            env.restore(copy.deepcopy(fixture))
            matched = _fingerprint(env) == fingerprint
            report["restore_checks"].append({"before": primitive, "valid": matched})
            if not matched:
                raise ValueError(f"fixture_restore_mismatch_before_{primitive}")
            if record_video:
                outcome = execute_recorded(
                    env,
                    replace(action, primitive=primitive),
                    config=c,
                    executor=execute_action,
                    output_dir=directory,
                    action_id=f"diagnostic-{primitive}",
                    appearance=appearance,
                    recorder_factory=recorder_factory,
                )
                report["recordings"].append(outcome.evidence["recording"])
                recording = outcome.evidence["recording"]
                if recording["recording_failed"] and recording["action_execution_valid"]:
                    env.release()
                    outcome = replace(outcome, after=env.observe())
            else:
                outcome = execute_action(env, replace(action, primitive=primitive), c)
            artifacts = (
                _save_observations(directory, primitive, outcome) if directory is not None else {}
            )
            report["runs"].append(
                {
                    "artifacts": artifacts,
                    "primitive": primitive,
                    "pixels": outcome.action.pixels,
                    "grasp_particle_ids": outcome.evidence.get("grasp_particle_ids", []),
                    "execution_valid": outcome.execution_valid,
                    "reason": outcome.reason,
                    "coverage_before": outcome.coverage_before,
                    "coverage_after": outcome.coverage_after,
                    "coverage_gain": outcome.coverage_after - outcome.coverage_before,
                    "area_before_m2": outcome.area_before_m2,
                    "area_after_m2": outcome.area_after_m2,
                    "simulated_seconds": outcome.simulated_seconds,
                    "wall_seconds": outcome.wall_seconds,
                    "diagnostics": outcome.diagnostics,
                    "evidence": outcome.evidence,
                }
            )
        report["execution_valid"] = all(r["execution_valid"] for r in report["runs"])
        if report["execution_valid"] and len(primitives) == 2:
            by_name = {r["primitive"]: r for r in report["runs"]}
            baseline, candidate = by_name["laydown"], by_name["fling"]
            report["candidate_minus_baseline_final_coverage"] = (
                candidate["coverage_after"] - baseline["coverage_after"]
            )
    except TimeoutError as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        report["execution_valid"] = False
        if hasattr(exc, "recording_evidence"):
            report["recordings"].append(exc.recording_evidence)
            try:
                env.release()
            except Exception as cleanup_exc:
                report["errors"].append({"stage": "release_cleanup", "message": str(cleanup_exc)})
        raise
    except Exception as exc:
        if hasattr(exc, "recording_evidence"):
            report["recordings"].append(exc.recording_evidence)
            try:
                env.release()
            except Exception as cleanup_exc:
                report["errors"].append({"stage": "release_cleanup", "message": str(cleanup_exc)})
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        report["execution_valid"] = False
    finally:
        if fixture is not None:
            try:
                env.restore(copy.deepcopy(fixture))
                report["final_restore_valid"] = _fingerprint(env) == fingerprint
                if not report["final_restore_valid"]:
                    raise ValueError("final_fixture_restore_mismatch")
            except TimeoutError:
                raise
            except Exception as exc:
                report["errors"].append(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "stage": "final_restore",
                    }
                )
                report["execution_valid"] = False
                report["final_restore_valid"] = False
        report["wall_seconds"] = time.monotonic() - started
        report = _json_safe(report)
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "diagnostic.json.tmp"
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(directory / "diagnostic.json")
    return report
