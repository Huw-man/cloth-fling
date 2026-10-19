"""CPU lifecycle checks; no physics or behavior qualification is implied."""

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from test_collection_pipeline import FakeEnv, execute, select
from test_primitives_fake import FakeClothBackend

from cloth_fling import cli, diagnostics
from cloth_fling.collection import collect_episodes
from cloth_fling.contracts import ActionOutcome, GraspAction
from cloth_fling.data import read_support, read_transitions, verify_dataset
from cloth_fling.provenance import config_provenance
from cloth_fling.sim.appearance import AppearanceConfig


class Recorder:
    instances = []
    callback_error = False
    close_error = False

    def __init__(self, output_dir, **kwargs):
        self.directory = Path(output_dir)
        self.kwargs = kwargs
        self.calls = self.closes = 0
        self.instances.append(self)

    def start_action(self, index):
        self.index = index

    def __call__(self, env, trace):
        self.calls += 1
        (self.directory / "combined.mp4").write_bytes(b"fake video; lifecycle test only")
        if self.callback_error:
            raise RuntimeError("capture failed")

    def close(self, *, complete):
        self.closes += 1
        metadata = {"complete": complete, "videos": [{"path": "combined.mp4", "sha256": "fake"}]}
        (self.directory / "video-metadata.json").write_text(json.dumps(metadata))
        if self.close_error:
            raise RuntimeError("encoder close failed")
        return metadata


@pytest.fixture(autouse=True)
def fresh_recorder():
    Recorder.instances = []
    Recorder.callback_error = Recorder.close_error = False


def recorded_executor(env, action, *, config, trace_callback):
    result = execute(env, action, config=config)
    trace_callback(env, {"phase": "terminal", "simulated_seconds": 0.4})
    return result


def test_recording_keeps_evolving_chain_and_unique_global_ids(tmp_path):
    env = FakeEnv()
    for _ in range(2):
        report = collect_episodes(
            env,
            tmp_path,
            max_actions=2,
            selector=select,
            executor=recorded_executor,
            record_video=True,
            recorder_factory=Recorder,
            appearance=AppearanceConfig(),
        )
        assert report["valid"] == 2
        assert len(report["recordings"]) == 2
        assert list(tmp_path.glob("collection-report-*.json"))
    assert len(env.resets) == 2 and not env.restores
    assert [r.index for r in Recorder.instances] == [
        "episode-0-action-0",
        "episode-0-action-1",
        "episode-1-action-0",
        "episode-1-action-1",
    ]
    assert all(r.closes == 1 for r in Recorder.instances)
    rows = read_transitions(tmp_path).to_pylist()
    assert rows[1]["before_rgb_path"] == rows[0]["after_rgb_path"]
    assert all(row["reward"] < 0 for row in rows)
    evidence = read_support(tmp_path, rows[0]["evidence_path"])["evidence"]["recording"]
    assert evidence["metadata"]["videos"][0]["relative_file"] == "combined.mp4"
    for artifact in evidence["files"]:
        assert (
            hashlib.sha256((tmp_path / artifact["relative_file"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert not any(name.startswith("video/") for name in manifest["files"])
    assert verify_dataset(tmp_path)["valid"]


@pytest.mark.parametrize("failure", ["callback", "close", "suppressed_callback"])
def test_capture_and_close_errors_invalidate_once_and_preserve_failed_video(tmp_path, failure):
    env = FakeEnv()
    Recorder.callback_error = failure != "close"
    Recorder.close_error = failure == "close"
    calls = []

    def executor(env, action, *, config, trace_callback):
        calls.append(action)
        env.attached_ids = (1, 2)
        result = execute(env, action, config=config)
        try:
            trace_callback(env, {})
        except RuntimeError:
            if failure != "suppressed_callback":
                raise
        return result

    report = collect_episodes(
        env,
        tmp_path,
        max_actions=2,
        selector=select,
        executor=executor,
        record_video=True,
        recorder_factory=Recorder,
    )
    assert len(calls) == 1 and env.release_calls == 1
    assert report["attempts"] == report["invalid"] == 1
    assert len(report["recordings"]) == 1 and report["recordings"][0]["files"]
    assert Recorder.instances[0].closes == 1
    row = read_transitions(tmp_path).to_pylist()[0]
    assert row["terminated"] and row["reward"] is None and row["after_observation_id"] == "1"
    assert verify_dataset(tmp_path)["valid"]


def test_no_recording_preserves_old_executor_signature_and_lazy_imports(tmp_path):
    code = """
import sys
from cloth_fling.cli import build_parser
from cloth_fling.collection import collect_episodes
from cloth_fling.diagnostics import run_diagnostic
build_parser().parse_args(['collect', '--output', 'unused'])
assert 'cloth_fling.rendering' not in sys.modules
assert 'imageio' not in sys.modules and 'imageio_ffmpeg' not in sys.modules
assert 'genesis' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    report = collect_episodes(FakeEnv(), tmp_path, max_actions=2, selector=select, executor=execute)
    assert report["valid"] == 2 and report["recordings"] == []
    assert not (tmp_path / "video").exists()


def test_diagnostic_exact_restore_recording_and_close_error(tmp_path, monkeypatch):
    env = FakeClothBackend()
    before = env.get_positions()
    executed = []
    Recorder.close_error = True

    def executor(env, action, *, config, trace_callback):
        executed.append(action.primitive)
        assert (env.get_positions() == before).all()
        observation = env.observe()
        trace_callback(env, {"phase": "before", "simulated_seconds": 0})
        return ActionOutcome(
            observation,
            observation,
            action,
            0.5,
            0.6,
            0.125,
            0.15,
            0.25,
            True,
            "ok",
            0.1,
            0.01,
        )

    monkeypatch.setattr(diagnostics, "execute_action", executor)
    report = diagnostics.run_diagnostic(
        env,
        output_dir=tmp_path,
        action=GraspAction(((25, 25), (38, 38))),
        record_video=True,
        recorder_factory=Recorder,
    )
    assert executed == ["laydown", "fling"]
    assert not report["execution_valid"] and report["final_restore_valid"]
    assert all(check["valid"] for check in report["restore_checks"])
    assert all(not run["execution_valid"] for run in report["runs"])
    assert len(report["recordings"]) == 2 and all(r.closes == 1 for r in Recorder.instances)
    assert (env.get_positions() == before).all()
    assert env.release_calls == 2


def test_appearance_provenance_is_independent_of_physics():
    env = FakeEnv()
    matte = config_provenance(env, appearance=AppearanceConfig())
    legacy = config_provenance(env, appearance=AppearanceConfig.legacy())
    assert matte["physics_config_sha256"] == legacy["physics_config_sha256"]
    assert matte["appearance_config_sha256"] != legacy["appearance_config_sha256"]
    parser = cli.build_parser()
    assert cli.load_appearance(parser.parse_args(["resets", "--output", "unused"])).mode == "matte"
    assert (
        cli.load_appearance(parser.parse_args(["replay", "unused", "--legacy-appearance"])).mode
        == "legacy"
    )


@pytest.mark.parametrize(
    "passed,half_dt,exit_code", [(True, True, 0), (False, True, 1), (True, False, 1)]
)
def test_contact_cli_dispatch_and_gate_exit(monkeypatch, tmp_path, passed, half_dt, exit_code):
    calls = []

    def run(config, **kwargs):
        calls.append((config, kwargs))
        kwargs["output_dir"].mkdir()
        return {"qualification_passed": passed, "gates": {"normal_energy": passed}}

    monkeypatch.setitem(
        sys.modules,
        "cloth_fling.contact_diagnostics",
        types.SimpleNamespace(run_contact_diagnostic=run),
    )
    path = tmp_path / "physics.json"
    path.write_text("{}")
    output = tmp_path / "contact"
    argv = [
        "diagnose-contact",
        "--config",
        str(path),
        "--output",
        str(output),
        "--duration-s",
        ".6",
    ]
    assert cli.main(argv + ([] if half_dt else ["--no-half-dt"])) == exit_code
    assert calls[0][1]["include_half_dt"] == half_dt
    assert (
        json.loads((output / "report.json").read_text())["full_contact_gate_requested"] == half_dt
    )


def test_replay_cli_actual_api_and_offline_inspection(monkeypatch, tmp_path):
    from cloth_fling import replay

    calls = []

    def render(source, output_dir, **kwargs):
        calls.append((source, output_dir, kwargs))
        return {"kind": "reconstructed_replay"}

    monkeypatch.setattr(replay, "render_replay", render)
    assert cli.main(["replay", "saved-dataset", "--output", str(tmp_path)]) == 0
    assert calls[0][2]["appearance"] == AppearanceConfig()
    assert calls[0][2]["max_frames"] == 10000
    monkeypatch.setattr(
        replay,
        "load_replay",
        lambda source, **kwargs: ([types.SimpleNamespace(duration=1)], {"actions": 1}),
    )
    assert cli.main(["replay", "old-diagnostic.json", "--inspect"]) == 0
    assert len(calls) == 1


def test_cli_simulation_appearance_and_invalid_exit(monkeypatch, tmp_path):
    from cloth_fling import collection
    from cloth_fling.sim import backend

    calls = []

    class Backend:
        def __init__(self, config, **kwargs):
            calls.append((config, kwargs))

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(backend, "GenesisBackend", Backend)
    monkeypatch.setattr(collection, "collect_episodes", lambda *args, **kwargs: {"invalid": 1})
    assert cli.main(["collect", "--output", str(tmp_path)]) == 1
    assert calls[0][1]["appearance"] == AppearanceConfig()
    assert calls[-1] == "closed"


@pytest.mark.parametrize("encoder_fails", [False, True])
def test_recorded_clearance_failure_keeps_motion_reason(tmp_path, encoder_fails):
    from test_primitives_clearance import HangingClothFake, selected

    from cloth_fling.provenance import execute_recorded
    from cloth_fling.sim.primitives import PrimitiveConfig, execute_action

    env = HangingClothFake(stays_supported=True)
    Recorder.close_error = encoder_fails
    result = execute_recorded(
        env,
        selected(env),
        config=PrimitiveConfig(clearance_max_height_m=0.4, tauten_seconds=0.2),
        executor=execute_action,
        output_dir=tmp_path,
        action_id="bounded-lift",
        recorder_factory=Recorder,
    )
    assert not result.execution_valid
    assert result.reason == "lift_max_height"
    assert result.diagnostics["clearance"]["failure_reason"] == "lift_max_height"
    recording = result.evidence["recording"]
    assert recording["action_reason"] == "lift_max_height"
    assert recording["action_execution_valid"] is False
    assert recording["recording_failed"] == encoder_fails
    if not encoder_fails:
        assert recording["metadata"]["complete"] is True
    assert Recorder.instances[0].calls > 1 and Recorder.instances[0].closes == 1
    assert env.attach_calls == 1 and env.release_calls == 1


def test_missing_video_encoder_is_explicit_invalid_attempt(tmp_path, monkeypatch):
    import numpy as np

    from cloth_fling import rendering

    env = FakeEnv()
    env.get_positions = lambda: env.snapshot()["positions"]
    monkeypatch.setattr(
        rendering,
        "render_debug_views",
        lambda *args, **kwargs: {
            name: np.zeros((64, 64, 3), dtype=np.uint8) for name in ("overhead", "oblique")
        },
    )
    monkeypatch.setitem(sys.modules, "imageio.v2", None)
    report = collect_episodes(
        env,
        tmp_path,
        max_actions=2,
        selector=select,
        executor=recorded_executor,
        record_video=True,
    )
    assert report["invalid"] == 1 and env.index == 1 and env.release_calls == 1
    recording = report["recordings"][0]
    assert "Install cloth-fling[video]" in recording["callback_error"]
    assert recording["recording_failed"] and not recording["metadata"]["complete"]
    assert verify_dataset(tmp_path)["valid"]


def test_recorded_diagnostic_timeout_preserves_report_and_restores(tmp_path, monkeypatch):
    env = FakeClothBackend()
    before = env.snapshot()

    def executor(env, action, *, config, trace_callback):
        trace_callback(env, {"phase": "before", "simulated_seconds": 0})
        raise TimeoutError("bounded execution interrupted")

    monkeypatch.setattr(diagnostics, "execute_action", executor)
    with pytest.raises(TimeoutError, match="bounded execution interrupted"):
        diagnostics.run_diagnostic(
            env,
            output_dir=tmp_path,
            action=GraspAction(((25, 25), (38, 38))),
            record_video=True,
            recorder_factory=Recorder,
        )
    report = json.loads((tmp_path / "diagnostic.json").read_text())
    assert not report["execution_valid"] and report["final_restore_valid"]
    assert report["recordings"][0]["recording_failed"]
    assert (env.get_positions() == before["positions"]).all()
    assert Recorder.instances[0].closes == 1


def test_runtime_and_native_video_canonical_config_hash_agree(tmp_path, monkeypatch):
    import numpy as np

    from cloth_fling import rendering

    env = FakeEnv()
    env.get_positions = lambda: env.snapshot()["positions"]
    appearance = AppearanceConfig()
    provenance = config_provenance(env, appearance=appearance)
    monkeypatch.setattr(
        rendering,
        "render_debug_views",
        lambda *args, **kwargs: {
            name: np.zeros((64, 64, 3), dtype=np.uint8) for name in ("overhead", "oblique")
        },
    )

    def writer_factory(path, **kwargs):
        Path(path).write_bytes(b"CPU fake encoder; config identity check")
        return types.SimpleNamespace(append_data=lambda frame: None, close=lambda: None)

    recorder = rendering.VideoRecorder(
        tmp_path, appearance=appearance, writer_factory=writer_factory
    )
    recorder.start_action("canonical-hash-check")
    recorder(env, {"phase": "before", "simulated_seconds": 0, "coverage": 0.8})
    metadata = recorder.close()
    assert list(metadata["physics_configs"]) == [provenance["physics_config_sha256"]]
    assert metadata["samples"][0]["physics_config_sha256"] == provenance["physics_config_sha256"]
    assert all(
        video["physics_config_sha256"] == [provenance["physics_config_sha256"]]
        for video in metadata["videos"]
    )
    assert (
        provenance["appearance_config_sha256"]
        == hashlib.sha256(
            json.dumps(
                appearance.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )
