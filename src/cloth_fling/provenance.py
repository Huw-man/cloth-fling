"""Read local code/runtime identity without importing the simulator or Torch."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def package_versions() -> dict[str, str | None]:
    """Distribution identity only; never imports the packages being inspected."""
    packages = {}
    for name in (
        "cloth-fling", "genesis-world", "quadrants", "torch", "trimesh", "numpy",
        "pyarrow", "pillow", "pydantic", "gs-madrona", "av", "imageio", "imageio-ffmpeg",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def runtime_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    packages = package_versions()
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "engine_version": packages["genesis-world"],
        "engine_version_source": "installed_distribution_metadata",
        "code_revision": None,
        "code_dirty": None,
        "dependency_lock_sha256": None,
        "torch_cuda_version": None,
        "gpu_summary": None,
    }
    if (root / ".git").exists():
        try:
            result["code_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            ).strip()
            result["code_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=root,
                    text=True,
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
    lock = root / "uv.lock"
    if lock.is_file():
        result["dependency_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if probe.returncode == 0:
            result["gpu_summary"] = probe.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def config_provenance(env, *, appearance=None, primitive_config=None, reset_source=None):
    """Independent render/physics settings; reset input identity includes array contents."""
    from dataclasses import asdict, is_dataclass

    from cloth_fling.data.storage import json_bytes, sha256

    def document(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return asdict(value) if is_dataclass(value) else value

    physics = document(env.config)
    rendering = document(appearance) if appearance is not None else {"mode": "legacy"}

    def config_hash(value):
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    return {
        "runtime_identity": document(getattr(env, "runtime_identity", None)),
        "backend_capabilities": document(getattr(env, "capabilities", None)),
        "rendering_identity": document(getattr(env, "rendering_identity", None)),
        "rendering_capabilities": document(getattr(env, "rendering_capabilities", None)),
        "contact_profile": getattr(env.config, "contact_profile", None),
        "source_snapshot_identity": (
            {
                key: reset_source.get(key)
                for key in (
                    "schema_version", "engine", "engine_version", "runtime_identity_json",
                    "config_json", "config_hash",
                )
            }
            if reset_source is not None else None
        ),
        "physics_config": physics,
        "physics_config_sha256": config_hash(physics),
        "appearance_config": rendering,
        "appearance_config_sha256": config_hash(rendering),
        "primitive_config": document(primitive_config),
        "source_reset_sha256": (
            sha256(json_bytes(reset_source)) if reset_source is not None else None
        ),
    }


def execute_recorded(
    env,
    action,
    *,
    config,
    executor,
    output_dir,
    action_id,
    appearance=None,
    recorder_factory=None,
):
    """Execute once, close once; capture failures invalidate rather than replay motion.

    Videos are external artifacts, intentionally not DatasetWriter registered references.
    Use ``relative_file`` instead of the verifier's reserved ``path``/``sha256`` pair.
    """
    from dataclasses import replace

    from cloth_fling.data.storage import file_sha256

    if recorder_factory is None:
        from cloth_fling.rendering import VideoRecorder

        recorder_factory = VideoRecorder
    root = Path(output_dir)
    directory = root / "video" / str(action_id)
    directory.mkdir(parents=True, exist_ok=False)
    recorder = None
    outcome = None
    failure = None
    close_failure = None
    metadata = {}
    callback_failure = None
    try:
        recorder = recorder_factory(
            directory,
            appearance=appearance,
            table_height_m=getattr(config, "table_height_m", 0.0),
            particle_radius_m=getattr(env, "particle_diameter", 0.0) / 2,
        )
        recorder.start_action(action_id)

        def callback(current_env, trace):
            nonlocal callback_failure
            try:
                return recorder(current_env, trace)
            except Exception as exc:
                callback_failure = f"{type(exc).__name__}: {exc}"
                raise

        outcome = executor(env, action, config=config, trace_callback=callback)
    except BaseException as exc:
        failure = exc
    finally:
        if recorder is not None:
            try:
                metadata = recorder.close(
                    complete=failure is None and callback_failure is None and outcome is not None
                )
            except Exception as exc:
                close_failure = f"{type(exc).__name__}: {exc}"

    def external(value):
        if isinstance(value, dict):
            return {
                "relative_file" if key == "path" else key: external(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [external(item) for item in value]
        return value

    artifacts, artifact_errors = [], []
    try:
        for candidate in sorted(directory.rglob("*")):
            if candidate.is_file():
                reference = candidate.relative_to(root).as_posix()
                try:
                    digest = file_sha256(candidate)
                except OSError as exc:
                    digest = None
                    artifact_errors.append(f"{reference}: {type(exc).__name__}: {exc}")
                artifacts.append({"relative_file": reference, "sha256": digest})
    except OSError as exc:
        artifact_errors.append(f"{type(exc).__name__}: {exc}")
    evidence = {
        "kind": "native_render_capture",
        "directory": directory.relative_to(root).as_posix(),
        "metadata": external(metadata),
        "callback_error": callback_failure,
        "close_error": close_failure,
        "execution_error": f"{type(failure).__name__}: {failure}" if failure is not None else None,
        "action_execution_valid": outcome.execution_valid if outcome is not None else None,
        "action_reason": outcome.reason if outcome is not None else None,
        "files": artifacts,
        "artifact_errors": artifact_errors,
        "dataset_manifest_registered": False,
    }
    empty_video = bool(metadata.get("videos")) and any(
        video.get("frames") == 0 or video.get("sha256") is None for video in metadata["videos"]
    )
    recording_failed = bool(
        failure
        or close_failure
        or callback_failure
        or empty_video
        or artifact_errors
        or not metadata.get("complete", False)
    )
    evidence["recording_failed"] = recording_failed
    if failure is not None:
        # Preserve timeout type and original error; collection journals terminal cleanup.
        failure.recording_evidence = evidence
        raise failure
    if recording_failed and outcome.execution_valid:
        outcome = replace(
            outcome,
            execution_valid=False,
            reason=f"recording_failed: {close_failure or callback_failure or artifact_errors or 'incomplete capture'}",
        )
    return replace(outcome, evidence={**outcome.evidence, "recording": evidence})
