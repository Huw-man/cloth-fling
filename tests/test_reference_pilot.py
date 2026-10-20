"""Integration guards with fake dynamics; native behavior is recorded separately."""

import json
import subprocess
import sys
import time
from contextlib import nullcontext

import numpy as np
import pytest
from test_frozen_laydown import HandoffFake

import cloth_fling.reference_pilot as pilot
from cloth_fling.data import read_support, read_transitions, verify_dataset
from cloth_fling.sim.backend import _state_hash
from cloth_fling.spatial_actions import SpatialGraspAction

ACTION = SpatialGraspAction(((1, 1), (0, 0)), encoding={"original": "unchanged"})


class Env(HandoffFake):
    def __init__(self):
        super().__init__()
        self.calibration = {}
        self.restore_calls = 0
        self.attach_calls = []

    def snapshot(self):
        state = {
            **super().snapshot(),
            "faces": self.faces.copy(),
            "rest_positions": self.rest_positions.copy(),
            "simulated_time": self.simulated_time,
            "config_json": json.dumps(self.config.to_dict()),
        }
        state["state_hash"] = _state_hash(state)
        return state

    def restore(self, state):
        self.restore_calls += 1
        self.p = state["positions"].copy()
        self.v = state["velocities"].copy()
        self.simulated_time = state["simulated_time"]
        return self.observe()

    def attach(self, pixels):
        self.attach_calls.append(pixels)
        return super().attach(pixels)

    def coverage(self):
        value = 0.8 - self.simulated_time * 0.001
        return value, value * 0.24, 0.24


@pytest.fixture
def resolution(monkeypatch):
    monkeypatch.setattr(pilot, "visible_particle", lambda pixel, *args: 20 if pixel[0] else 0)


def test_resolution_orders_execution_without_rewriting_original_action(resolution):
    env = Env()
    result = pilot.resolve_grasp(env, ACTION)
    assert result["resolved_native_ids"] == (20, 0)
    assert result["execution_native_ids"] == [0, 20]
    assert result["execution_permutation"] == [1, 0]
    assert ACTION.pixels == ((1, 1), (0, 0))
    assert result["action_encoding"] == {"original": "unchanged"}
    assert env.attach_calls == [] and env.simulated_time == 0


def test_two_actions_adopt_once_chain_exactly_and_keep_negative_gain(
    tmp_path, monkeypatch, resolution
):
    env = Env()
    fixture = env.snapshot()
    selected, rng_values = [], []

    def select(obs, config, *, rng, **kwargs):
        selected.append(obs)
        rng_values.append(int(rng.integers(100000)))
        return ACTION

    monkeypatch.setattr(pilot, "select_spatial_action", select)
    report = pilot.collect_pilot(env, tmp_path, fixture, time.monotonic() + 60)
    assert report["attempts"] == report["valid"] == 2, report
    assert env.restore_calls == 1 and len(env.attach_calls) == 2
    assert all(pixels == ((0, 0), (1, 1)) for pixels in env.attach_calls)
    expected = np.random.default_rng(100)
    assert rng_values == [int(expected.integers(100000)) for _ in range(2)]
    rows = read_transitions(tmp_path / "dataset").to_pylist()
    assert rows[1]["before_observation_id"] == selected[1].observation_id
    assert rows[1]["before_rgb_path"] == rows[0]["after_rgb_path"]
    assert rows[1]["before_support_path"] == rows[0]["after_support_path"]
    assert rows[0]["coverage_gain"] < 0 and rows[1]["coverage_gain"] < 0
    assert verify_dataset(tmp_path / "dataset")["valid"]
    for i, row in enumerate(rows):
        assert json.loads((tmp_path / f"action-{i:02d}-cleanup.json").read_text())["passed"]
        evidence = read_support(tmp_path / "dataset", row["evidence_path"])["evidence"]
        assert evidence["retained_existing_grasp"]
        assert evidence["image_grasp_handoff"]["execution_permutation"] == [1, 0]
    assert pilot.assess(tmp_path)["integration_passed"]


@pytest.mark.parametrize("failure", ["duplicate", "partial_attach", "handoff", "cleanup"])
def test_failures_stop_without_reroll_and_certify_cleanup(
    tmp_path, monkeypatch, resolution, failure
):
    env = Env()
    fixture = env.snapshot()
    calls = []
    monkeypatch.setattr(pilot, "select_spatial_action", lambda *a, **k: calls.append(1) or ACTION)
    if failure == "duplicate":
        monkeypatch.setattr(pilot, "visible_particle", lambda *a: 0)
    elif failure == "partial_attach":

        def attach(pixels):
            env.attached_ids = (0,)
            raise RuntimeError("partial native acquisition")

        env.attach = attach
    elif failure == "handoff":
        monkeypatch.setattr(pilot, "execute_frozen_laydown", lambda *a, **k: 1 / 0)
    else:

        def release():
            raise RuntimeError("cleanup failure")

        env.release = release
        monkeypatch.setattr(pilot, "execute_frozen_laydown", lambda *a, **k: 1 / 0)
    report = pilot.collect_pilot(env, tmp_path, fixture, time.monotonic() + 60)
    assert report["attempts"] == report["invalid"] == 1
    assert calls == [1]
    certificate = json.loads((tmp_path / "action-00-cleanup.json").read_text())
    assert certificate["passed"] is (failure != "cleanup")
    assert (tmp_path / "action-00-failure.json").exists()
    assert not (tmp_path / "action-01-intent.json").exists()
    assert not pilot.assess(tmp_path)["integration_passed"]


def test_restore_regression_stops_before_selection(tmp_path, monkeypatch):
    env = Env()
    fixture = env.snapshot()
    restore = env.restore

    def drift(state):
        result = restore(state)
        env.p[0, 0] += 1e-6
        return result

    env.restore = drift
    monkeypatch.setattr(pilot, "select_spatial_action", lambda *a, **k: pytest.fail("selected"))
    report = pilot.collect_pilot(env, tmp_path, fixture, time.monotonic() + 60)
    assert report["reset_failures"] == 1 and report["attempts"] == 0
    assert not json.loads((tmp_path / "restoration.json").read_text())["passed"]


@pytest.mark.parametrize("status", ["started_never_resume", "failed_or_uncertain", "complete"])
def test_existing_experiment_cannot_open_scene_or_retry(tmp_path, status):
    root = tmp_path / "claimed"
    root.mkdir()
    (root / "pilot-report.json").write_text(json.dumps({"status": status}))
    with pytest.raises(FileExistsError):
        pilot.run_pilot(root, tmp_path / "missing", backend_factory=lambda *a: pytest.fail("scene"))


def test_uncertain_attempt_claim_cannot_be_reexecuted(tmp_path, resolution):
    env = Env()
    executor = pilot.PilotExecutor(tmp_path, time.monotonic() + 10)
    (tmp_path / "action-00-intent.json").write_text("{}")
    with pytest.raises(FileExistsError):
        executor(env, ACTION, config=None)
    assert not env.attach_calls


def test_runner_imports_without_genesis_or_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import cloth_fling.reference_pilot; import sys; "
            "assert 'genesis' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("render_failure", [False, True])
def test_run_claims_once_closes_scene_and_preserves_actions_on_render_failure(
    tmp_path, monkeypatch, resolution, render_failure
):
    env = Env()
    env.close = lambda: setattr(env, "closed", True)
    fixture = env.snapshot()
    source = tmp_path / "fixture.json"
    source.write_text("{}")
    monkeypatch.setattr(pilot, "FIXTURE_SHA256", pilot.file_sha256(source))
    monkeypatch.setattr(pilot, "FIXTURE_STATE_HASH", fixture["state_hash"])
    monkeypatch.setattr(pilot, "read_hashed_bundle", lambda *a: fixture)
    monkeypatch.setattr(pilot, "reference_lease", lambda *a: nullcontext())
    monkeypatch.setattr(pilot, "select_spatial_action", lambda *a, **k: ACTION)
    rendered = []

    def render(source, target, **kwargs):
        assert env.closed
        assert verify_dataset(source)["valid"]
        rendered.append((source, target, kwargs))
        if render_failure:
            raise RuntimeError("encoder failed")
        return {"truncated": False, "physics_reexecuted": False}

    root = tmp_path / "run"
    predecessor = tmp_path / "prior-report.json"
    predecessor.write_text(json.dumps({"status": "complete", "wall_seconds": 105.8}))
    if render_failure:
        with pytest.raises(RuntimeError, match="encoder failed"):
            pilot.run_pilot(
                root,
                source,
                backend_factory=lambda config: env,
                renderer=render,
                budget_predecessor=predecessor,
            )
        report = json.loads((root / "pilot-report.json").read_text())
        assert report["status"] == "failed_or_uncertain" and not report["integration_passed"]
    else:
        report = pilot.run_pilot(
            root,
            source,
            backend_factory=lambda config: env,
            renderer=render,
            budget_predecessor=predecessor,
        )
        assert report["status"] == "complete" and report["integration_passed"]
    assert len(rendered) == 1 and rendered[0][2]["fps"] == 10
    assert report["cumulative_wall_seconds"] == pytest.approx(105.8 + report["wall_seconds"])
    assert len(report["actions"]) == 2
    assert all(a["cleanup_passed"] for a in report["actions"])
    assert verify_dataset(root / "dataset")["valid"]
    with pytest.raises(FileExistsError):
        pilot.run_pilot(root, source, backend_factory=lambda *a: pytest.fail("scene"))
