"""Milestone 3 storage/lifecycle proofs with injected CPU fakes, never simulation."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_collection_pipeline import FakeEnv, execute, select

from cloth_fling.collection import collect_episodes
from cloth_fling.data import read_support, read_transitions, verify_dataset
from cloth_fling.data.storage import json_bytes, sha256


class CoverageEnv(FakeEnv):
    def __init__(self, initial=0.7, speed=0.0):
        super().__init__()
        self.initial = initial
        self.speed = speed

    def reset(self, seed=0, *, kind="drop"):
        super().reset(seed, kind=kind)
        self.value = self.initial
        return self.observe()

    def coverage(self):
        return self.value, self.value * 0.25, 0.25

    def snapshot(self):
        return {**super().snapshot(), "velocities": np.full((2, 3), self.speed)}


def advancing_executor(values, *, invalid_last=False):
    values = iter(values)

    def executor(env, action, *, config, trace_callback=None):
        result = execute(env, action, config=config)
        env.value = next(values)
        if trace_callback:
            trace_callback(env, {"phase": "terminal", "simulated_seconds": 0.4})
        return replace(
            result,
            coverage_after=env.value,
            area_after_m2=env.value * 0.25,
            execution_valid=not (invalid_last and env.index == 2),
            reason="strain_guard" if invalid_last and env.index == 2 else "ok",
            diagnostics={"settled": True, "terminal_observation_valid": True},
        )

    return executor


def manifest_and_summaries(root):
    manifest = json.loads((root / "manifest.json").read_text())
    return manifest, [
        read_support(root, episode["collection_summary"]["path"])
        for episode in manifest["episodes"]
    ]


def test_two_two_action_episodes_select_fresh_results_without_intermediate_reset(tmp_path):
    env = CoverageEnv()
    selected, emitted = [], []
    executor = advancing_executor([0.6, 0.8, 0.65, 0.85])

    def fresh_selector(observation, config, **kwargs):
        if env.index:
            assert observation is emitted[-1].after
        assert len(env.resets) == len(selected) // 2 + 1
        assert not env.restores
        selected.append(observation)
        return select(observation, config, **kwargs)

    def wrapped_executor(env, action, **kwargs):
        result = executor(env, action, **kwargs)
        emitted.append(result)
        return result

    report = collect_episodes(
        env,
        tmp_path,
        episodes=2,
        max_actions=2,
        selector=fresh_selector,
        executor=wrapped_executor,
        settled_coverage_threshold=0.95,
        split="validation",
        reset_group_id="fresh-bunch-family-7",
        cloth_id="terry-1",
        episode_metadata={"benchmark": "m3", "initial_settled": False},
    )
    assert len(env.resets) == 2 and not env.restores and env.release_calls == 0
    assert report["valid"] == report["attempts"] == 4
    assert report["invalid"] == 0
    rows = read_transitions(tmp_path).to_pylist()
    for offset in (0, 2):
        previous, following = rows[offset : offset + 2]
        assert following["before_rgb_path"] == previous["after_rgb_path"]
        assert following["before_support_path"] == previous["after_support_path"]
        assert following["before_observation_id"] == previous["after_observation_id"]
        assert previous["coverage_gain"] < 0
        assert following["end_reason"] == "max_actions" and following["truncated"]
    manifest, summaries = manifest_and_summaries(tmp_path)
    assert summaries == report["episode_reports"]
    for episode, summary in zip(manifest["episodes"], summaries):
        assert episode["split"] == summary["split"] == "validation"
        assert episode["reset_group_id"] == "fresh-bunch-family-7"
        assert episode["cloth_id"] == "terry-1"
        assert summary["initial_confirmed_settled_coverage"] == 0.7
        assert summary["lifecycle"] == "stopped_without_threshold"
    assert [s["last_confirmed_settled_coverage"] for s in summaries] == [0.8, 0.85]
    assert verify_dataset(tmp_path)["valid"]


def test_threshold_reached_on_valid_settled_fresh_result(tmp_path):
    env = CoverageEnv()
    report = collect_episodes(
        env,
        tmp_path,
        max_actions=5,
        selector=select,
        executor=advancing_executor([0.8, 0.95]),
        settled_coverage_threshold=0.95,
    )
    assert env.index == report["valid"] == 2
    row = read_transitions(tmp_path).to_pylist()[-1]
    assert row["end_reason"] == "settled_coverage_threshold"
    assert row["terminated"] and not row["truncated"]
    summary = report["episode_reports"][0]
    assert summary["lifecycle"] == "unfolded_after_action"
    assert summary["last_confirmed_settled_coverage"] == 0.95
    assert summary["last_confirmed_settled_observation_id"] == row["after_observation_id"]
    assert verify_dataset(tmp_path)["valid"]


def test_invalid_above_threshold_geometry_does_not_replace_confirmed_coverage(tmp_path):
    report = collect_episodes(
        CoverageEnv(),
        tmp_path,
        max_actions=5,
        selector=select,
        executor=advancing_executor([0.8, 0.99], invalid_last=True),
        settled_coverage_threshold=0.95,
    )
    summary = report["episode_reports"][0]
    assert report["valid"] == report["invalid"] == 1
    assert summary["end_reason"] == "execution_invalid"
    assert summary["last_observed_coverage"] == 0.99
    assert summary["last_observed_execution_valid"] is False
    assert summary["last_confirmed_settled_coverage"] == 0.8
    assert summary["last_confirmed_settled_observation_id"] == "1"
    rows = read_transitions(tmp_path).to_pylist()
    assert rows[-1]["coverage_after"] == 0.99 and rows[-1]["reward"] is None
    assert rows[-1]["after_observation_id"] == "2"
    assert verify_dataset(tmp_path)["valid"]


@pytest.mark.parametrize("kind", ["unsettled", "unknown_settlement", "stale_observation"])
def test_valid_unconfirmed_observation_cannot_trigger_threshold(tmp_path, kind):
    executor = advancing_executor([0.99])

    def unconfirmed(env, action, **kwargs):
        outcome = executor(env, action, **kwargs)
        if kind == "stale_observation":
            return replace(outcome, after=outcome.before)
        return replace(outcome, diagnostics={"settled": False} if kind == "unsettled" else {})

    report = collect_episodes(
        CoverageEnv(),
        tmp_path,
        max_actions=1,
        selector=select,
        executor=unconfirmed,
        settled_coverage_threshold=0.95,
    )
    summary = report["episode_reports"][0]
    assert summary["end_reason"] == "max_actions"
    assert summary["last_confirmed_settled_coverage"] == 0.7
    assert summary["last_observed_coverage"] == 0.99
    assert verify_dataset(tmp_path)["valid"]


@pytest.mark.parametrize(
    "restored,certified,speed,stops",
    [
        (False, False, 0.0, True),
        (True, True, 0.0, True),
        (True, False, 0.0, False),
        (True, True, 0.02, False),
        (False, False, 0.02, False),
    ],
)
def test_initial_threshold_requires_settled_reset_without_added_steps(
    tmp_path, restored, certified, speed, stops
):
    env = CoverageEnv(initial=0.98, speed=speed)
    saved = {"index": 0, "value": 0.98} if restored else None
    report = collect_episodes(
        env,
        tmp_path,
        max_actions=1,
        selector=select,
        executor=execute,
        reset_state=saved,
        episode_metadata={"initial_settled": certified},
        settled_coverage_threshold=0.95,
    )
    summary = report["episode_reports"][0]
    assert len(env.restores) == int(restored) and len(env.resets) == int(not restored)
    assert env.index == report["attempts"] == int(not stops)
    assert summary["initial_coverage"] == 0.98
    if stops:
        assert summary["lifecycle"] == summary["end_reason"] == "initially_unfolded"
        assert summary["last_confirmed_settled_coverage"] == 0.98
        assert read_transitions(tmp_path).num_rows == 0
    else:
        assert summary["initial_confirmed_settled_coverage"] is None
        assert summary["last_confirmed_settled_coverage"] is None
    assert verify_dataset(tmp_path)["valid"]


def test_metadata_deep_copied_per_episode_and_reset_failure_keeps_identities(tmp_path):
    env = CoverageEnv()
    metadata = {"case": {"name": "pilot"}, "split": "caller note; not identity"}
    original_reset = env.reset

    def reset(seed, *, kind):
        if env.resets:
            raise RuntimeError("reset failed")
        return original_reset(seed, kind=kind)

    def selector(observation, config, **kwargs):
        metadata["case"]["name"] = "mutated outside collector"
        return select(observation, config, **kwargs)

    env.reset = reset
    report = collect_episodes(
        env,
        tmp_path,
        episodes=2,
        max_actions=1,
        selector=selector,
        executor=execute,
        split="test",
        reset_group_id="explicit-family",
        cloth_id="cloth-2",
        episode_metadata=metadata,
    )
    manifest, summaries = manifest_and_summaries(tmp_path)
    assert report["reset_failures"] == 1
    for episode, summary in zip(manifest["episodes"], summaries):
        assert episode["split"] == "test"
        assert episode["reset_group_id"] == "explicit-family"
        assert episode["cloth_id"] == "cloth-2"
        assert episode["metadata"]["case"]["name"] == "pilot"
        assert summary["metadata"] == episode["metadata"]
    assert summaries[-1]["end_reason"] == "reset_failed"
    assert summaries[-1]["initial_confirmed_settled_coverage"] is None
    assert verify_dataset(tmp_path)["valid"]


def test_new_run_metadata_does_not_relabel_existing_episodes(tmp_path):
    for name in ("first", "second"):
        collect_episodes(
            CoverageEnv(),
            tmp_path,
            max_actions=0,
            selector=select,
            executor=execute,
            episode_metadata={"case": name},
        )
    manifest, _ = manifest_and_summaries(tmp_path)
    assert [e["metadata"]["case"] for e in manifest["episodes"]] == ["first", "second"]
    assert verify_dataset(tmp_path)["valid"]


@pytest.mark.parametrize("threshold", [0, -1, 1.1, True, "0.95", float("nan"), float("inf")])
def test_threshold_validation_precedes_physical_calls(tmp_path, threshold):
    env = CoverageEnv()
    with pytest.raises(ValueError, match="settled_coverage_threshold"):
        collect_episodes(env, tmp_path, settled_coverage_threshold=threshold)
    assert not env.resets and not env.restores


@pytest.mark.parametrize("identity", ["split", "reset_group_id", "cloth_id"])
def test_identity_validation_precedes_physical_calls(tmp_path, identity):
    env = CoverageEnv()
    with pytest.raises(ValueError, match=identity):
        collect_episodes(env, tmp_path, **{identity: ""})
    assert not env.resets


def test_verifier_checks_metadata_against_immutable_reset_support(tmp_path):
    collect_episodes(CoverageEnv(), tmp_path, max_actions=0, episode_metadata={"case": "pilot"})
    manifest, _ = manifest_and_summaries(tmp_path)
    manifest["episodes"][0]["metadata"]["case"] = "rewritten"
    (tmp_path / "manifest.json").write_bytes(json_bytes(manifest))
    result = verify_dataset(tmp_path)
    assert not result["valid"]
    assert any("episode metadata mismatch" in error for error in result["errors"])


def test_verifier_rejects_promoting_invalid_geometry_even_with_rehashed_summary(tmp_path):
    collect_episodes(
        CoverageEnv(),
        tmp_path,
        max_actions=5,
        selector=select,
        executor=advancing_executor([0.8, 0.99], invalid_last=True),
        settled_coverage_threshold=0.95,
    )
    manifest, summaries = manifest_and_summaries(tmp_path)
    ref = manifest["episodes"][0]["collection_summary"]
    summaries[0]["last_confirmed_settled_coverage"] = 0.99
    content = json_bytes({"payload": summaries[0], "arrays": None})
    (tmp_path / ref["path"]).write_bytes(content)
    ref["sha256"] = sha256(content)
    manifest["files"][ref["path"]] = {"sha256": sha256(content), "size_bytes": len(content)}
    (tmp_path / "manifest.json").write_bytes(json_bytes(manifest))
    result = verify_dataset(tmp_path)
    assert not result["valid"]
    assert any("last confirmed settled coverage mismatch" in error for error in result["errors"])


class CpuRecorder:
    def __init__(self, directory, **kwargs):
        self.directory = directory
        self.calls = 0

    def start_action(self, action_id):
        self.action_id = action_id

    def __call__(self, env, trace):
        self.calls += 1
        (self.directory / "combined.mp4").write_bytes(b"CPU lifecycle placeholder")

    def close(self, *, complete):
        return {"complete": complete}


def test_recorded_and_unrecorded_timings_storage_and_first_paired_split_cases(tmp_path):
    for split in ("train", "test"):
        for recorded in (True, False):
            root = tmp_path / split / str(recorded)
            report = collect_episodes(
                CoverageEnv(),
                root,
                max_actions=2,
                selector=select,
                executor=advancing_executor([0.8, 0.85]),
                split=split,
                reset_group_id=f"{split}-pilot",
                episode_metadata={"paired_case": 0},
                record_video=recorded,
                recorder_factory=CpuRecorder,
            )
            mode = "recorded" if recorded else "unrecorded"
            other = "unrecorded" if recorded else "recorded"
            timing = report["timings"]["by_recording"]
            assert timing[mode]["attempts"] == 2 and timing[mode]["execution_seconds"] > 0
            assert timing[mode]["simulated_seconds"] == 0.8
            assert timing[other]["attempts"] == timing[other]["execution_seconds"] == 0
            assert report["timings"]["storage_seconds"] > 0
            storage = report["storage"]
            assert storage["registered_bytes_total"] >= storage["registered_bytes_added"] > 0
            assert not storage["external_recording_stat_errors"]
            assert storage["external_recording_file_count"] == (2 if recorded else 0)
            assert storage["external_recording_bytes"] == (
                2 * len(b"CPU lifecycle placeholder") if recorded else 0
            )
            assert len(report["recordings"]) == (2 if recorded else 0)
            assert verify_dataset(root)["valid"]
