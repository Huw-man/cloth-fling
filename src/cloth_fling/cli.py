"""Small command surface. Offline commands do not initialize simulation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from cloth_fling import __version__
from cloth_fling.contracts import SceneConfig


def load_config(path: str | None) -> SceneConfig:
    if path is None:
        return SceneConfig()
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("scene configuration must be a JSON object")
    unknown = set(data) - {field.name for field in fields(SceneConfig)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    return SceneConfig(**data)


def doctor_report() -> dict[str, Any]:
    packages = {}
    for name in ("cloth-fling", "genesis-world", "torch", "numpy", "pyarrow", "pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpu = {
            "available": probe.returncode == 0,
            "summary": probe.stdout.strip(),
            "error": probe.stderr.strip() if probe.returncode else None,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu = {"available": False, "summary": "", "error": str(exc)}
    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
        "simulation_probed": False,
    }


def load_reset(path: Path) -> dict[str, Any]:
    """Read a local reset without pickle or simulator initialization."""
    import numpy as np

    metadata = json.loads((path / "state.json").read_text())
    if not isinstance(metadata, dict):
        raise ValueError("reset metadata must be a JSON object")
    with np.load(path / "state.npz", allow_pickle=False) as archive:
        if set(metadata) & set(archive.files):
            raise ValueError("reset has conflicting array and metadata keys")
        return {**metadata, **{key: archive[key].copy() for key in archive.files}}


def _backend(args: argparse.Namespace):
    from cloth_fling.sim.backend import GenesisBackend

    return GenesisBackend(load_config(args.config), backend=args.backend, headless=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robotless cloth simulation and episode datasets")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect dependencies/GPU; optionally build a scene")
    doctor.add_argument(
        "--probe", action="store_true", help="Build and render a flat Genesis scene"
    )

    resets = sub.add_parser("resets", help="Generate a bounded reset and save its state")
    resets.add_argument("--output", type=Path, required=True)
    resets.add_argument("--kind", choices=("flat", "drop", "fold"), default="fold")
    resets.add_argument("--seed", type=int, default=0)

    diagnose = sub.add_parser("diagnose", help="Compare primitives on one restored fixture")
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--seed", type=int, default=0)
    diagnose.add_argument("--primitive-config", help="Separate motion/settling JSON overrides")
    diagnose.add_argument("--reset-kind", choices=("flat", "drop", "fold"), default="fold")
    diagnose.add_argument("--reset-dir", type=Path, help="Compare from a saved fixture")
    diagnose.add_argument(
        "--grasp-pixels",
        type=float,
        nargs=4,
        metavar=("U0", "V0", "U1", "V1"),
        help="Use a manually reviewed favorable grasp pair for both primitives",
    )

    collect = sub.add_parser("collect", help="Collect bounded evolving-cloth episodes")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--episodes", type=int, default=1)
    collect.add_argument("--max-actions", type=int, default=3)
    collect.add_argument("--max-wall-seconds", type=float, default=600)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--primitive-config", help="Separate motion/settling JSON overrides")
    collect.add_argument(
        "--reset-dir", type=Path, help="Restore a saved reset at episode boundaries"
    )
    collect.add_argument("--reset-kind", choices=("flat", "drop", "fold"), default="fold")
    collect.add_argument(
        "--policy", choices=("uniform_valid", "geometry_guided"), default="uniform_valid"
    )
    collect.add_argument("--primitive", choices=("laydown", "fling"), default="laydown")

    for command in (doctor, resets, diagnose, collect):
        command.add_argument("--config", help="Scene JSON overrides")
        command.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")

    dataset = sub.add_parser("dataset", help="Offline dataset inspection")
    data_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    verify = data_sub.add_parser(
        "verify", help="Verify Parquet, references, hashes and episode links"
    )
    verify.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dataset":
            from cloth_fling.data import verify_dataset

            report = verify_dataset(args.path)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report.get("valid", report.get("ok", True)) else 1
        if args.command == "doctor":
            report = doctor_report()
            if args.probe:
                env = _backend(args)
                try:
                    obs = env.reset(kind="flat")
                    report["simulation_probed"] = True
                    report["scene"] = {
                        "rgb_shape": list(obs.rgb.shape),
                        "coverage": list(env.coverage()),
                        "attached_ids": list(env.attached_ids),
                    }
                finally:
                    env.close()
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "collect":
            if args.episodes < 1 or not 1 <= args.max_actions <= 10:
                raise ValueError("episodes must be positive and max-actions must be in [1, 10]")
            if not 0 < args.max_wall_seconds <= 86400:
                raise ValueError("max-wall-seconds must be in (0, 86400]")
        primitive_config = None
        if args.command in ("collect", "diagnose") and args.primitive_config:
            from cloth_fling.sim.primitives import PrimitiveConfig

            primitive_config = PrimitiveConfig(
                **json.loads(Path(args.primitive_config).read_text())
            )
        env = _backend(args)
        try:
            if args.command == "collect":
                from cloth_fling.collection import collect_episodes

                report = collect_episodes(
                    env,
                    args.output,
                    episodes=args.episodes,
                    max_actions=args.max_actions,
                    seed=args.seed,
                    policy=args.policy,
                    primitive=args.primitive,
                    max_wall_seconds=args.max_wall_seconds,
                    reset_state=load_reset(args.reset_dir) if args.reset_dir else None,
                    reset_kind=args.reset_kind,
                    primitive_config=primitive_config,
                )
            elif args.command == "diagnose":
                from cloth_fling.contracts import GraspAction
                from cloth_fling.diagnostics import run_diagnostic

                if args.reset_dir:
                    env.restore(load_reset(args.reset_dir))
                else:
                    env.reset(args.seed, kind=args.reset_kind)
                action = None
                if args.grasp_pixels:
                    u0, v0, u1, v1 = args.grasp_pixels
                    action = GraspAction(((u0, v0), (u1, v1)), policy="manual_favorable")
                report = run_diagnostic(
                    env,
                    seed=args.seed,
                    output_dir=args.output,
                    action=action,
                    primitive_config=primitive_config,
                )
            else:
                import numpy as np
                from PIL import Image

                args.output.mkdir(parents=True, exist_ok=False)
                observation = env.reset(args.seed, kind=args.kind)
                state = env.snapshot()
                arrays = {
                    key: value for key, value in state.items() if isinstance(value, np.ndarray)
                }
                metadata = {key: value for key, value in state.items() if key not in arrays}
                np.savez_compressed(args.output / "state.npz", **arrays)
                (args.output / "state.json").write_text(json.dumps(metadata, indent=2))
                Image.fromarray(observation.rgb).save(args.output / "rgb.png")
                (args.output / "camera.json").write_text(
                    json.dumps(observation.calibration, indent=2)
                )
                report = {
                    "output": str(args.output),
                    "seed": args.seed,
                    "kind": args.kind,
                    "coverage": list(env.coverage()),
                }
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        finally:
            env.close()
    except (ValueError, TypeError, RuntimeError, OSError, ImportError) as exc:
        print(f"cloth-fling: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
