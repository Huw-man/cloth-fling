"""Replay compatibility and no-physics guarantees using tiny real dataset supports."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from test_rendering import Encoder, Env

from cloth_fling.contracts import ActionOutcome, GraspAction, Observation, SceneConfig
from cloth_fling.data import DatasetWriter
from cloth_fling.diagnostics import _save_fixture
from cloth_fling.replay import (
    ReplayAction,
    load_replay,
    main,
    render_recorded_state,
    render_replay,
    sample_traces,
    source_engine_identity,
    validate_action,
)
from cloth_fling.sim.backend import _state_hash


def snapshot():
    state = {
        "schema_version": 1,
        "faces": np.array([[0, 1, 2]], dtype=np.int32),
        "rest_positions": Env().p.copy(),
        "positions": Env().p.copy(),
        "velocities": Env().v.copy(),
        "free": np.ones(3, dtype=bool),
        "simulated_time": 0.0,
        "config_json": json.dumps(SceneConfig(image_size=16).to_dict()),
    }
    state["state_hash"] = _state_hash(state)
    return state


def traces():
    p = Env().p
    return [
        {
            "phase": phase,
            "simulated_seconds": t,
            "attachment_ids": ids,
            "cloth_positions_m": p + [0, 0, dz],
            "cloth_velocities_m_s": np.zeros_like(p),
            "coverage": coverage,
        }
        for phase, t, ids, dz, coverage in [
            ("before", 0.0, [], 0, 0.8),
            ("lift", 0.1, [0, 1], 0.2, 0.6),
            ("released", 0.2, [], 0, 0.4),
        ]
    ]


def dataset(root, *, valid=True, with_traces=True):
    before = Observation("before", np.full((16, 16, 3), 100, dtype=np.uint8))
    after = Observation("after", np.full((16, 16, 3), 120, dtype=np.uint8))
    result = ActionOutcome(
        before,
        after,
        GraspAction(((1.0, 2.0), (3.0, 4.0))),
        0.8,
        0.4,
        0.2,
        0.1,
        0.25,
        valid,
        "completed" if valid else "invalid_attempt",
        0.2,
        0.1,
        terminated=not valid,
        truncated=valid,
        evidence={"phase_traces": traces() if with_traces else []},
    )
    with DatasetWriter(root) as writer:
        eid = writer.save_reset(before, seed=0, config=SceneConfig(image_size=16), state=snapshot())
        writer.append_outcome(eid, result, final_state=snapshot())


def diagnostic(root):
    root.mkdir()
    fixture = _save_fixture(root, snapshot())
    document = {
        "schema": "cloth_fling.diagnostic.v1",
        "fixture_artifacts": fixture,
        "runs": [
            {
                "primitive": "laydown",
                "execution_valid": False,
                "reason": "diagnostic_failure",
                "simulated_seconds": 0.2,
                "evidence": {"phase_traces": traces()},
            }
        ],
    }
    # Original diagnostic format serializes traces to nested JSON lists.
    path = root / "diagnostic.json"
    path.write_text(json.dumps(document, default=lambda v: v.tolist()))
    return path


def test_old_dataset_loads_invalid_attempts_and_negative_coverage(tmp_path):
    root = tmp_path / "data"
    dataset(root, valid=False)
    actions, origin = load_replay(root)
    assert origin["format"] == "dataset" and origin["invalid_attempts"] == 1
    assert not actions[0].execution_valid
    assert actions[0].traces[-1]["coverage"] < actions[0].traces[0]["coverage"]
    assert origin["source_manifest_sha256"]
    assert origin["source_engines"][0]["version"] == "0.4.3"
    assert origin["runtime_qualification_inherited"] is False
    with pytest.raises(ValueError, match="no replay actions"):
        load_replay(root, episode_id=99)


def test_old_diagnostic_fixture_arrays_decode_without_pickle(tmp_path):
    path = diagnostic(tmp_path / "diagnostic")
    actions, origin = load_replay(path)
    assert origin["format"] == "diagnostic" and len(origin["fixture_hashes"]) == 2
    np.testing.assert_array_equal(actions[0].state["positions"], snapshot()["positions"])
    assert actions[0].reason == "diagnostic_failure"
    assert main([str(path), "--inspect"]) == 0


def test_empty_trajectory_is_explicit_terminal_only(tmp_path):
    root = tmp_path / "data"
    dataset(root, valid=False, with_traces=False)
    actions, _ = load_replay(root)
    assert actions[0].traces[0]["phase"] == "terminal_only_no_trajectory"
    assert not actions[0].traces[0]["attachment_ids"]


def test_hold_sampling_uses_past_trace_and_terminal_without_interpolation():
    action = validate_action(ReplayAction("one", snapshot(), traces(), 0.2, True, "ok", {}, {}))
    samples = list(sample_traces(action, 20))
    assert [s["phase"] for s in samples] == ["before", "before", "lift", "lift", "released"]
    assert samples[1]["source_trace_seconds"] == 0
    np.testing.assert_array_equal(samples[1]["cloth_positions_m"], traces()[0]["cloth_positions_m"])
    assert samples[-1]["simulated_seconds"] == 0.2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda t: t[0].update(simulated_seconds=-1),
        lambda t: t[0].update(cloth_positions_m=[[0, 0, 0]]),
        lambda t: t[1].update(simulated_seconds=float("nan")),
        lambda t: t[2].update(simulated_seconds=0.05),
        lambda t: t[1].update(attachment_ids=[0, 4]),
    ],
)
def test_malformed_trajectory_rejected_before_simulator(mutation):
    t = traces()
    mutation(t)
    with pytest.raises(ValueError):
        validate_action(ReplayAction("one", snapshot(), t, 0.2, True, "ok", {}, {}))


def test_render_only_import_preserves_source_and_records_old_engine():
    state = snapshot()
    old = copy.deepcopy(state)
    env = ReplayEnv(SceneConfig(image_size=16))
    render_recorded_state(env, state, traces()[1])
    render_recorded_state(env, state, traces()[2])
    assert env.geometry_imports == 1
    np.testing.assert_array_equal(env.p, traces()[2]["cloth_positions_m"])
    assert state["state_hash"] == old["state_hash"]
    for key in ("positions", "free", "rest_positions", "faces"):
        np.testing.assert_array_equal(state[key], old[key])
    assert source_engine_identity(state)["version"] == "0.4.3"


class ReplayEnv(Env):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.attached_ids = ()
        self.particle_diameter = 0.01
        self.faces = snapshot()["faces"].copy()
        self.rest_positions = snapshot()["rest_positions"].copy()
        from types import SimpleNamespace

        self.cloth = SimpleNamespace(set_particles_pos=self.set_recorded_positions)
        self.geometry_imports = 0

    def restore(self, state):
        raise AssertionError("physical snapshot restore forbidden in geometry replay")

    def restore_recorded_geometry(self, state):
        self.geometry_imports += 1
        # If replay calls a physics hook, the installed guard must fail.
        for obj in (self, self.scene):
            if hasattr(obj, "step"):
                with pytest.raises(RuntimeError, match="never step"):
                    obj.step()
        self.p = state["positions"].copy()

    def set_recorded_positions(self, positions):
        self.p = positions.copy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


@pytest.mark.parametrize("key", ["faces", "rest_positions"])
def test_topology_mismatch_or_missing_recorded_geometry_fails_before_import(key):
    state = snapshot()
    env = ReplayEnv(SceneConfig(image_size=16))
    env.restore_recorded_geometry = lambda *a, **kw: pytest.fail(
        "must not import mismatched geometry"
    )
    state[key] = state[key].copy()
    state[key].flat[0] += 1
    with pytest.raises(ValueError, match=f"{key} mismatch"):
        render_recorded_state(env, state, traces()[0])
    del state[key]
    with pytest.raises(ValueError, match=f"requires recorded {key}"):
        render_recorded_state(env, state, traces()[0])


def test_no_fallback_to_physical_restore_and_exact_import_readback():
    env = ReplayEnv(SceneConfig(image_size=16))
    env.restore_recorded_geometry = None
    with pytest.raises(RuntimeError, match="physical restore is forbidden"):
        render_recorded_state(env, snapshot(), traces()[0])
    env.restore_recorded_geometry = lambda *a, **kw: None
    env.cloth.set_particles_pos = lambda *a: None
    with pytest.raises(RuntimeError, match="readback differs"):
        render_recorded_state(env, snapshot(), traces()[1])


def test_new_schema_requires_explicit_engine_identity():
    with pytest.raises(ValueError, match="missing its source engine"):
        source_engine_identity({"schema_version": 2})
    identity = {"engine": "genesis", "engine_version": "1.4.4", "physics_schema_version": 2}
    result = source_engine_identity(
        {"schema_version": 2, "runtime_identity_json": json.dumps(identity)}
    )
    assert result["backend_identity"] == identity
    assert result["version"] == "1.4.4"


def test_default_factory_uses_new_template_and_keeps_archival_source_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import cloth_fling.replay as replay
    import cloth_fling.sim.backend as backend

    root = tmp_path / "data"
    dataset(root)

    # Model the runtime-owned v2 contract without changing contracts in this
    # isolated rendering branch. The integrated contract accepts v2 directly.
    class TemplateConfig(SimpleNamespace):
        def to_dict(self):
            return vars(self).copy()

    monkeypatch.setattr(replay, "SceneConfig", lambda **kw: TemplateConfig(**kw))
    monkeypatch.setattr(backend, "GenesisBackend", ReplayEnv)
    metadata = render_replay(root, tmp_path / "video", writer_factory=Encoder)
    assert all(
        c["contact_profile"] == "native_coulomb_v2"
        for c in metadata["render_template_configs"].values()
    )
    assert all(
        c.get("contact_profile", "legacy_mg_v1") == "legacy_mg_v1"
        for c in metadata["source_configs"].values()
    )
    assert metadata["source_engines"][0]["version"] == "0.4.3"


def test_full_replay_cpu_framewriter_metadata_and_dataset_immutable(tmp_path):
    root, output = tmp_path / "data", tmp_path / "video"
    dataset(root)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    metadata = render_replay(
        root, output, fps=20, backend_factory=ReplayEnv, writer_factory=Encoder
    )
    assert metadata["kind"] == "reconstructed_replay"
    assert metadata["physics_reexecuted"] is False
    assert metadata["captured_during_execution"] is False
    assert metadata["original_video_recorded"] is False
    assert metadata["source_engines"][0]["version"] == "0.4.3"
    assert metadata["renderer_engines"][0]["version"] == "1.4.4"
    assert metadata["runtime_qualification_inherited"] is False
    for video in metadata["videos"]:
        assert video["renderer_engines"] == metadata["renderer_engines"]
        assert video["source_engines"] == metadata["source_engines"]
    assert metadata["videos"][-1]["frames"] == 5
    assert metadata["videos"][-1]["dimensions"] == [768, 384]
    assert len(metadata["original_observations"]) == 2
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    with pytest.raises(FileExistsError):
        render_replay(root, output, backend_factory=ReplayEnv)
    with pytest.raises(ValueError, match="outside"):
        render_replay(root, root / "video", backend_factory=ReplayEnv)


def test_max_frames_is_explicit_truncation(tmp_path):
    root = tmp_path / "data"
    dataset(root)
    metadata = render_replay(
        root, tmp_path / "video", max_frames=2, backend_factory=ReplayEnv, writer_factory=Encoder
    )
    assert metadata["truncated"] and metadata["videos"][-1]["frames"] == 2


def test_reference_dataset_uses_wide_cameras_without_executing_physics(tmp_path, monkeypatch):
    from cloth_fling.reference_surface import diagonal_scene

    state = snapshot()
    state["config_json"] = json.dumps(diagonal_scene().to_dict())
    state["state_hash"] = _state_hash(state)
    monkeypatch.setattr(sys.modules[__name__], "snapshot", lambda: state)
    root = tmp_path / "data"
    dataset(root)
    metadata = render_replay(
        root, tmp_path / "video", backend_factory=ReplayEnv, writer_factory=Encoder
    )
    assert metadata["view_poses"]["overhead"]["pos"] == (0.55, 0, 2.9)
    assert metadata["view_poses"]["oblique"]["pos"] == (1.9, -1.9, 1.7)
    assert metadata["physics_reexecuted"] is False


def test_import_and_inspection_work_with_sim_video_dependencies_blocked(tmp_path):
    path = diagnostic(tmp_path / "diagnostic")
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'genesis', 'torch', 'imageio', 'imageio_ffmpeg'}:
            raise RuntimeError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
import cloth_fling.sim.render_mesh
from cloth_fling.replay import main
assert main([sys.argv[1], '--inspect']) == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_failure_returns_nonzero(tmp_path, capsys):
    assert main([str(tmp_path / "missing"), "--inspect"]) == 1
    assert "replay failed" in capsys.readouterr().err
