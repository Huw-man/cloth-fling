"""Read local code/runtime identity without importing the simulator or Torch."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def runtime_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {
        "platform": platform.platform(),
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
