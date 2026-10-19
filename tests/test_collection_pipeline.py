import json
import sys
import time
import types
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from cloth_fling.collection import collect_episodes
from cloth_fling.contracts import ActionOutcome, GraspAction, Observation, SceneConfig
from cloth_fling.data import read_support, read_transitions, verify_dataset


class FakeEnv:
    config = SceneConfig(image_size=16)

    def __init__(self):
        self.resets = []
        self.restores = []
        self.index = 0
        self.value = 0.8
        self.attached_ids = ()
        self.release_calls = 0

    def reset(self, seed=0, *, kind="drop"):
        self.resets.append((seed, kind))
        self.index = 0
        self.value = 0.8
        return self.observe()

    def restore(self, state):
        self.restores.append(state)
        self.index = state["index"]
        self.value = state["value"]
        return self.observe()

    def observe(self):
        return Observation(
            str(self.index),
            np.full((4, 6, 3), self.index, dtype=np.uint8),
            np.ones((4, 6)),
            {"world_z_up": True},
        )

    def snapshot(self):
        return {
            "index": self.index,
            "value": self.value,
            "attached_ids": self.attached_ids,
            "positions": np.full((2, 3), self.value),
            "velocities": np.zeros((2, 3)),
        }

    def release(self):
        self.attached_ids = ()
        self.release_calls += 1


def select(observation, config, *, policy, primitive, rng):
    return GraspAction(((0.5, 1.5), (4, 2.0)), primitive, policy)


def execute(env, action, *, config):
    before = env.observe()
    initial = env.value
    env.index += 1
    env.value -= 0.1
    return ActionOutcome(
        before,
        env.observe(),
        action,
        initial,
        env.value,
        initial * 0.25,
        env.value * 0.25,
        0.25,
        True,
        "ok",
        0.4,
        0.01,
    )


def test_consecutive_actions_chain_exact_objects_without_reset(tmp_path):
    env = FakeEnv()
    selected = []
    emitted = []

    def selector(obs, config, **kwargs):
        if emitted:
            assert obs is emitted[-1].after
        selected.append(obs)
        return select(obs, config, **kwargs)

    def executor(env, action, *, config):
        assert config == {"speed": 2}
        result = execute(env, action, config=config)
        emitted.append(result)
        return result

    report = collect_episodes(
        env,
        tmp_path,
        max_actions=3,
        selector=selector,
        executor=executor,
        primitive_config={"speed": 2},
        seed=2,
    )
    assert len(env.resets) == 1 and env.index == 3
    assert report["attempts"] == report["valid"] == 3 and report["invalid"] == 0
    assert report["episodes"] == 1 and len(selected) == 3
    rows = read_transitions(tmp_path).to_pylist()
    for previous, current in pairwise(rows):
        assert current["before_observation_id"] == previous["after_observation_id"]
        assert current["before_rgb_path"] == previous["after_rgb_path"]
        assert current["before_support_path"] == previous["after_support_path"]
    assert not rows[0]["truncated"] and rows[-1]["truncated"]
    assert rows[-1]["end_reason"] == "max_actions"
    assert all(row["reward"] < 0 for row in rows)
    assert report["timings"]["wall_seconds"] > 0
    verified = verify_dataset(tmp_path)
    assert verified["valid"], verified


def test_invalid_execution_recorded_before_new_episode_reset(tmp_path):
    env = FakeEnv()

    def executor(env, action, *, config):
        env.attached_ids = (1, 2)
        return replace(
            execute(env, action, config=config), execution_valid=False, reason="overstretch"
        )

    report = collect_episodes(
        env, tmp_path, episodes=2, max_actions=4, selector=select, executor=executor
    )
    assert report["attempts"] == report["invalid"] == 2 and report["valid"] == 0
    assert len(env.resets) == 2 and env.release_calls == 2
    rows = read_transitions(tmp_path).to_pylist()
    assert all(row["reward"] is None and row["terminated"] for row in rows)
    assert all(row["end_reason"] == "execution_invalid" for row in rows)
    assert verify_dataset(tmp_path)["valid"]


def test_execution_exception_records_terminal_observation_and_cleanup(tmp_path):
    env = FakeEnv()

    def executor(env, action, *, config):
        env.index = 1
        env.attached_ids = (3, 8)
        raise RuntimeError("simulated fault")

    report = collect_episodes(env, tmp_path, selector=select, executor=executor)
    row = read_transitions(tmp_path).to_pylist()[0]
    assert report["invalid"] == 1 and env.release_calls == 1
    assert row["after_observation_id"] == "1" and row["reward"] is None
    assert "simulated fault" in row["reason"]
    assert read_support(tmp_path, row["final_state_path"])["attached_ids"] == []
    assert verify_dataset(tmp_path)["valid"]


def test_cleanup_error_is_evidence(tmp_path):
    env = FakeEnv()

    def release():
        raise RuntimeError("release failed")

    def executor(env, action, *, config):
        raise RuntimeError("execution failed")

    env.release = release
    report = collect_episodes(env, tmp_path, selector=select, executor=executor)
    row = read_transitions(tmp_path).to_pylist()[0]
    evidence = read_support(tmp_path, row["evidence_path"])
    assert "release failed" in evidence["cleanup"]["release_error"]
    assert report["invalid"] == 1


@pytest.mark.parametrize(
    "selection",
    [lambda *a, **kw: None, lambda *a, **kw: (_ for _ in ()).throw(ValueError("no pair"))],
)
def test_no_valid_pair_zero_step_metadata(tmp_path, selection):
    report = collect_episodes(FakeEnv(), tmp_path, selector=selection, executor=execute)
    assert report["episodes"] == 1 and report["attempts"] == 0
    assert report["episode_reports"][0]["end_reason"] == "no_valid_pair"
    assert read_transitions(tmp_path).num_rows == 0
    assert verify_dataset(tmp_path)["valid"]


def test_selection_exhaustion_after_a_valid_action(tmp_path):
    def selector(obs, config, **kwargs):
        return None if obs.observation_id == "1" else select(obs, config, **kwargs)

    report = collect_episodes(FakeEnv(), tmp_path, selector=selector, executor=execute)
    assert report["attempts"] == 1
    verified = verify_dataset(tmp_path)
    assert verified["valid"], verified


def test_timeout_interrupts_executor_and_records_no_retry(tmp_path):
    env = FakeEnv()
    calls = []

    def executor(env, action, *, config):
        calls.append(action)
        env.attached_ids = (2, 4)
        time.sleep(1.5)
        raise AssertionError("deadline failed")

    report = collect_episodes(
        env, tmp_path, selector=select, executor=executor, max_wall_seconds=0.2
    )
    assert len(calls) == 1 and env.release_calls == 1 and len(env.resets) == 1
    assert report["invalid"] == 1 and report["stop_reason"] == "max_wall_seconds"
    assert report["timings"]["wall_seconds"] < 1.5
    row = read_transitions(tmp_path).to_pylist()[0]
    assert row["truncated"] and row["reason"] == "execution_timeout"
    assert verify_dataset(tmp_path)["valid"]


def test_invalid_reset_preserves_previous_steps_and_stops_explicitly(tmp_path):
    env = FakeEnv()
    original = env.reset

    def reset(seed, *, kind):
        if env.resets:
            raise ValueError("invalid reset preparation")
        return original(seed, kind=kind)

    env.reset = reset
    report = collect_episodes(
        env, tmp_path, episodes=3, max_actions=2, selector=select, executor=execute
    )
    assert report["episodes"] == 2 and report["reset_failures"] == 1
    assert report["attempts"] == 2 and report["stop_reason"] == "reset_failed"
    assert report["timings"]["wall_seconds"] > 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["episodes"][-1]["initial_observation"] is None
    assert manifest["episodes"][-1]["steps"] == 0
    verified = verify_dataset(tmp_path)
    assert verified["valid"], verified


def test_resume_fresh_episode_and_saved_reset_group(tmp_path):
    env = FakeEnv()
    saved = {"index": 0, "value": 0.6, "attached_ids": [], "config": {"foo": 2}}
    first = collect_episodes(
        env, tmp_path, selector=select, executor=execute, max_actions=2, reset_state=saved
    )
    second = collect_episodes(
        env, tmp_path, selector=select, executor=execute, max_actions=2, reset_state=saved
    )
    assert not env.resets and len(env.restores) == 2
    assert first["run_id"] != second["run_id"]
    rows = read_transitions(tmp_path).to_pylist()
    assert [row["episode_id"] for row in rows] == [0, 0, 1, 1]
    assert rows[2]["parent_run_id"] == rows[0]["run_id"]
    assert rows[0]["reset_group_id"] == rows[2]["reset_group_id"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    reset = read_support(tmp_path, manifest["episodes"][0]["reset"]["path"])
    assert reset["reset_source"] == saved
    assert reset["state"]["value"] == 0.6
    assert verify_dataset(tmp_path)["valid"]


@pytest.mark.parametrize("max_actions", [-1, 11, 1.5, True])
def test_action_limit_validated(tmp_path, max_actions):
    with pytest.raises(ValueError):
        collect_episodes(
            FakeEnv(), tmp_path, selector=select, executor=execute, max_actions=max_actions
        )


def test_zero_actions_still_stores_episode_reset(tmp_path):
    report = collect_episodes(
        FakeEnv(), tmp_path, selector=select, executor=execute, max_actions=0, reset_kind="flat"
    )
    assert report["episodes"] == 1 and report["attempts"] == 0
    assert verify_dataset(tmp_path)["valid"]


def test_reset_timeout_is_separate_from_action_timeout(tmp_path):
    env = FakeEnv()

    def slow_reset(seed, *, kind):
        time.sleep(1)

    env.reset = slow_reset
    report = collect_episodes(
        env, tmp_path, selector=select, executor=execute, max_wall_seconds=0.2
    )
    assert report["stop_reason"] == "reset_timeout"
    assert report["reset_failures"] == 1 and report["attempts"] == 0
    assert read_transitions(tmp_path).num_rows == 0
    assert verify_dataset(tmp_path)["valid"]


def test_initial_snapshot_failure_is_stored_as_failed_preparation(tmp_path):
    env = FakeEnv()

    def broken_snapshot():
        raise ValueError("unavailable reset state")

    env.snapshot = broken_snapshot
    report = collect_episodes(env, tmp_path, selector=select, executor=execute)
    assert report["reset_failures"] == 1 and report["attempts"] == 0
    assert report["stop_reason"] == "reset_failed"
    assert verify_dataset(tmp_path)["valid"]


def test_default_lazy_dependency_signatures(tmp_path, monkeypatch):
    selection = types.ModuleType("cloth_fling.selection")
    selection.select_action = select
    sim = types.ModuleType("cloth_fling.sim")
    sim.__path__ = []
    primitives = types.ModuleType("cloth_fling.sim.primitives")
    primitives.execute_action = execute
    monkeypatch.setitem(sys.modules, "cloth_fling.selection", selection)
    monkeypatch.setitem(sys.modules, "cloth_fling.sim", sim)
    monkeypatch.setitem(sys.modules, "cloth_fling.sim.primitives", primitives)
    report = collect_episodes(FakeEnv(), tmp_path, max_actions=2)
    assert report["valid"] == 2
    assert verify_dataset(tmp_path)["valid"]
