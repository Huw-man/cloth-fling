"""Small command surface. Offline commands do not initialize simulation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from cloth_fling import __version__
from cloth_fling.contracts import ReferenceSceneConfig, SceneConfig, scene_config_from_dict


def load_config(path: str | None) -> SceneConfig:
    if path is None:
        return SceneConfig(contact_profile="native_coulomb_v2")
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("scene configuration must be a JSON object")
    cls = (
        ReferenceSceneConfig
        if data.get("profile") in ("reference_laydown_v1", "reference_diagonal_v3")
        else SceneConfig
    )
    unknown = set(data) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    return scene_config_from_dict(data)


def doctor_report() -> dict[str, Any]:
    from cloth_fling.provenance import package_versions

    packages = package_versions()
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
        "expected_engine_version": "1.4.4",
        "engine_version_matches_pin": (
            packages["genesis-world"] == "1.4.4" if packages["genesis-world"] else None
        ),
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

    config = load_config(args.config)
    if args.profile == "reference":
        if args.config and not isinstance(config, ReferenceSceneConfig):
            raise ValueError("reference profile conflicts with square configuration")
        config = config if args.config else ReferenceSceneConfig()
    factory = GenesisBackend
    if isinstance(config, ReferenceSceneConfig):
        from cloth_fling.sim.reference import ReferenceBackend

        factory = ReferenceBackend
    return factory(
        config,
        backend=args.backend,
        headless=True,
        appearance=load_appearance(args),
    )


def load_appearance(args):
    """CLI defaults to matte; backend/library None remains legacy."""
    from cloth_fling.sim.appearance import AppearanceConfig

    if args.legacy_appearance:
        return AppearanceConfig.legacy()
    if args.appearance_config:
        return AppearanceConfig.from_json(args.appearance_config)
    return AppearanceConfig()


def _appearance_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--appearance-config", help="Independent render-only appearance JSON")
    group.add_argument("--legacy-appearance", action="store_true", help="Original table appearance")


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
    resets.add_argument("--kind", choices=("flat", "drop", "fold", "authored"), default="fold")
    resets.add_argument("--seed", type=int, default=0)

    diagnose = sub.add_parser("diagnose", help="Compare primitives on one restored fixture")
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--seed", type=int, default=0)
    diagnose.add_argument("--primitive-config", help="Separate motion/settling JSON overrides")
    diagnose.add_argument(
        "--reset-kind", choices=("flat", "drop", "fold", "authored"), default="fold"
    )
    diagnose.add_argument("--reset-dir", type=Path, help="Compare from a saved fixture")
    diagnose.add_argument("--record-video", action="store_true", help="Native debug captures")
    diagnose.add_argument(
        "--primitives", nargs="+", choices=("laydown", "fling"), default=("laydown", "fling")
    )
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
    collect.add_argument("--record-video", action="store_true", help="Native debug captures")
    collect.add_argument(
        "--reset-dir", type=Path, help="Restore a saved reset at episode boundaries"
    )
    collect.add_argument(
        "--reset-kind", choices=("flat", "drop", "fold", "authored"), default="fold"
    )
    collect.add_argument(
        "--policy", choices=("uniform_valid", "geometry_guided"), default="uniform_valid"
    )
    collect.add_argument("--primitive", choices=("laydown", "fling"), default="laydown")
    collect.add_argument("--split")
    collect.add_argument("--reset-group-id")
    collect.add_argument("--cloth-id")
    collect.add_argument("--episode-metadata", type=Path, help="Episode metadata JSON object")
    collect.add_argument("--settled-coverage-threshold", type=float)

    reference = sub.add_parser("reference", help="Bounded reference towel qualification")
    reference.add_argument("stage", choices=("qualify", "tune", "contact-qualify", "diagonal"))
    reference.add_argument(
        "--fixture", type=Path, help="Archived failed reference campaign (tune/contact-qualify)"
    )
    reference.add_argument(
        "--contact-report",
        type=Path,
        help="Pinned contact report in the same campaign (diagonal only)",
    )
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument(
        "--previous", type=Path, help="Carry all charges from a terminal setup attempt"
    )
    reference.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")

    learning = sub.add_parser(
        "learning-pilot", help="Bounded RGB grasp learning with frozen laydown"
    )
    learning.add_argument(
        "stage_command", choices=("prepare", "collect", "train", "evaluate", "report", "run")
    )
    learning.add_argument(
        "--stage",
        choices=(
            "prepare",
            "exploration",
            "initial",
            "refresh",
            "refreshed",
            "validation",
            "test",
            "replays",
        ),
    )
    learning.add_argument("--output", type=Path, default=Path("runs/image-learning-pilot-v1"))

    overfit = sub.add_parser("learning-overfit", help="Overfit one reset within three actions")
    overfit.add_argument("stage_command", choices=("run", "report"))
    overfit.add_argument("--output", type=Path, default=Path("runs/single-reset-overfit-v1"))

    benchmark = sub.add_parser("benchmark", help="Bounded milestone-3 benchmark")
    benchmark.add_argument("stage", choices=("prepare", "qualify", "pilot", "report"))
    benchmark.add_argument("--output", type=Path, default=Path("runs/milestone3"))
    benchmark.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    benchmark.add_argument(
        "--revision-checks", action="store_true", help="Run the single frozen revision's checks"
    )
    benchmark.add_argument(
        "--revision-config", type=Path, help="Measured targeted laydown revision configuration"
    )
    benchmark.add_argument("--revision-reason", help="Measured failure addressed by revision")

    for command in (doctor, resets, diagnose, collect):
        command.add_argument("--profile", choices=("square", "reference"), default="square")
        command.add_argument("--config", help="Scene JSON; default: native_coulomb_v2")
        command.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
        _appearance_options(command)

    contact = sub.add_parser("diagnose-contact", help="Bounded native table-contact qualification")
    contact.add_argument("--config", required=True, help="Native-contact SceneConfig JSON")
    contact.add_argument("--output", type=Path, required=True)
    contact.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    contact.add_argument("--duration-s", type=float, default=0.6)
    contact.add_argument(
        "--no-half-dt", action="store_true", help="Incomplete gate; exploratory only"
    )

    replay = sub.add_parser("replay", help="Reconstruct saved dataset/diagnostic trajectories")
    replay.add_argument("source", type=Path)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--episode-id", type=int)
    replay.add_argument("--fps", type=float, default=20)
    replay.add_argument("--max-frames", type=int, default=10000)
    replay.add_argument("--inspect", action="store_true", help="Offline validation only")
    _appearance_options(replay)

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
        if args.command == "learning-overfit":
            from cloth_fling.learning_overfit import main as overfit_main

            return overfit_main([args.stage_command, "--output", str(args.output)])
        if args.command == "learning-pilot":
            from cloth_fling.learning_pilot import main as learning_main

            forwarded = [args.stage_command, "--output", str(args.output)]
            if args.stage is not None:
                forwarded += ["--stage", args.stage]
            return learning_main(forwarded)
        if args.command == "reference":
            from cloth_fling.reference_run import run_reference_qualification

            if args.contact_report is not None and args.stage != "diagonal":
                raise ValueError("--contact-report applies to reference diagonal only")
            if args.stage in ("contact-qualify", "diagonal"):
                from cloth_fling.reference_diagonal_run import (
                    run_contact_qualification,
                    run_diagonal_campaign,
                )

                if args.previous is not None:
                    raise ValueError("new diagonal campaign does not accept --previous")
                if args.stage == "contact-qualify":
                    if args.fixture is None or args.contact_report is not None:
                        raise ValueError("contact-qualify requires --fixture only")
                    report = run_contact_qualification(
                        args.output, fixture_dir=args.fixture, backend=args.backend
                    )
                else:
                    if args.fixture is not None:
                        raise ValueError("diagonal uses the fixture pinned by contact-qualify")
                    report = run_diagonal_campaign(
                        args.output, contact_report=args.contact_report, backend=args.backend
                    )
            elif args.stage == "tune":
                from cloth_fling.reference_tune import run_reference_tuning

                if args.fixture is None or args.previous is not None:
                    raise ValueError(
                        "reference tune requires --fixture and does not accept --previous"
                    )
                report = run_reference_tuning(
                    args.output, fixture_dir=args.fixture, backend=args.backend
                )
            else:
                if args.fixture is not None:
                    raise ValueError("--fixture applies to reference tune only")
                report = run_reference_qualification(
                    args.output, backend=args.backend, previous=args.previous
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report["stage_passed"] else 1
        if args.command == "benchmark":
            from cloth_fling.benchmark_run import run_benchmark_stage

            report = run_benchmark_stage(
                args.stage,
                args.output,
                backend=args.backend,
                revision_checks=args.revision_checks,
                revision_config=args.revision_config,
                revision_reason=args.revision_reason,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report.get("stage_passed", report.get("complete", False)) else 1
        if args.command == "dataset":
            from cloth_fling.data import verify_dataset

            report = verify_dataset(args.path)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report.get("valid", report.get("ok", True)) else 1
        if args.command == "replay":
            from cloth_fling.replay import load_replay, render_replay

            if args.inspect:
                actions, report = load_replay(args.source, episode_id=args.episode_id)
                report["durations_seconds"] = [a.duration for a in actions]
            else:
                if args.output is None:
                    raise ValueError("replay requires --output unless --inspect is set")
                report = render_replay(
                    args.source,
                    args.output,
                    fps=args.fps,
                    episode_id=args.episode_id,
                    appearance=load_appearance(args),
                    max_frames=args.max_frames,
                )
                from cloth_fling.provenance import runtime_provenance

                report["runtime_provenance"] = runtime_provenance()
                (args.output / "replay-provenance.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "diagnose-contact":
            from cloth_fling.contact_diagnostics import run_contact_diagnostic

            if (args.output / "report.json").exists():
                raise FileExistsError("contact report exists; choose a fresh output directory")
            if not 0.3 <= args.duration_s <= 2:
                raise ValueError("duration-s must be in [0.3, 2]")
            report = run_contact_diagnostic(
                load_config(args.config),
                output_dir=args.output,
                backend=args.backend,
                duration_s=args.duration_s,
                include_half_dt=not args.no_half_dt,
            )
            from cloth_fling.provenance import runtime_provenance

            report["runtime_provenance"] = runtime_provenance()
            report["full_contact_gate_requested"] = not args.no_half_dt
            if args.no_half_dt:
                report["qualification_passed"] = False
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report.get("qualification_passed") and not args.no_half_dt else 1
        if args.command == "doctor":
            report = doctor_report()
            if args.probe:
                env = _backend(args)
                try:
                    obs = env.reset(
                        kind="authored" if isinstance(env.config, ReferenceSceneConfig) else "flat"
                    )
                    report["simulation_probed"] = True
                    report["scene"] = {
                        "rgb_shape": list(obs.rgb.shape),
                        "coverage": list(env.coverage()),
                        "attached_ids": list(env.attached_ids),
                    }
                    from cloth_fling.provenance import config_provenance, runtime_provenance

                    report["provenance"] = config_provenance(env, appearance=load_appearance(args))
                    report["runtime_provenance"] = runtime_provenance()
                finally:
                    env.close()
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "collect":
            if args.episodes < 1 or not 1 <= args.max_actions <= 10:
                raise ValueError("episodes must be positive and max-actions must be in [1, 10]")
            if not 0 < args.max_wall_seconds <= 86400:
                raise ValueError("max-wall-seconds must be in (0, 86400]")
        if args.command in ("collect", "diagnose") and (
            args.profile == "reference"
            or isinstance(load_config(args.config), ReferenceSceneConfig)
        ):
            raise ValueError("use 'reference qualify' for the bounded reference campaign")
        primitive_config = None
        if args.command in ("collect", "diagnose"):
            from cloth_fling.sim.primitives import PrimitiveConfig

            primitive_config = PrimitiveConfig(
                **(
                    json.loads(Path(args.primitive_config).read_text())
                    if args.primitive_config
                    else {}
                )
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
                    record_video=args.record_video,
                    appearance=load_appearance(args),
                    split=args.split,
                    reset_group_id=args.reset_group_id,
                    cloth_id=args.cloth_id,
                    episode_metadata=(
                        json.loads(args.episode_metadata.read_text())
                        if args.episode_metadata
                        else None
                    ),
                    settled_coverage_threshold=args.settled_coverage_threshold,
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
                    record_video=args.record_video,
                    appearance=load_appearance(args),
                    primitives=tuple(args.primitives),
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
                from cloth_fling.provenance import config_provenance, runtime_provenance

                report["provenance"] = config_provenance(
                    env,
                    appearance=load_appearance(args),
                    reset_source=state,
                )
                report["runtime_provenance"] = runtime_provenance()
                (args.output / "reset-provenance.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            if args.command == "diagnose":
                return 0 if report.get("execution_valid") else 1
            return (
                1
                if args.command == "collect"
                and (report.get("reset_failures", 0) or report.get("invalid", 0))
                else 0
            )
        finally:
            env.close()
    except (ValueError, TypeError, KeyError, RuntimeError, OSError, ImportError) as exc:
        print(f"cloth-fling: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
