import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cloth_fling.contracts import ActionOutcome, GraspAction, Observation, SceneConfig
from cloth_fling.data import (
    DatasetWriter,
    read_rgb,
    read_support,
    read_transitions,
    transition_schema,
    verify_dataset,
)
from cloth_fling.data.storage import relative_path


def observation(index):
    return Observation(
        str(index),
        np.full((5, 7, 3), index, dtype=np.uint8),
        np.full((5, 7), 0.5 + index, dtype=np.float32),
        {"intrinsics": np.eye(3), "world_z_up": True},
    )


def outcome(before, after, *, valid=True, final=False):
    return ActionOutcome(
        before,
        after,
        GraspAction(((1.25, 2), (4, 3.5))),
        0.8,
        0.4,
        0.2,
        0.1,
        0.25,
        valid,
        "ok" if valid else "bad_state",
        1.0,
        0.01,
        truncated=final,
        evidence={"trace": np.arange(4)},
    )


def make_dataset(root, *, state=None, valid=True):
    before, after = observation(0), observation(1)
    with DatasetWriter(root, {"split": "train"}) as writer:
        eid = writer.save_reset(
            before,
            seed=7,
            config=SceneConfig(),
            state={"positions": np.zeros((2, 3)), "attached_ids": [], "seed": 7},
        )
        result = outcome(before, after, valid=valid, final=True)
        writer.begin_attempt(eid, result.action, before=before)
        row = writer.append_outcome(
            eid,
            result,
            final_state=state
            or {
                "attached_ids": np.array([], dtype=np.int64),
                "positions": np.ones((2, 3)),
                "velocities": np.zeros((2, 3)),
            },
        )
    return row


def rewrite_part(root, mutate):
    manifest = json.loads((root / "manifest.json").read_text())
    part = manifest["parts"][0]
    path = root / part["path"]
    row = pq.read_table(path).to_pylist()[0]
    mutate(row)
    pq.write_table(pa.Table.from_pylist([row], schema=transition_schema()), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    part["sha256"] = digest
    manifest["files"][part["path"]].update(sha256=digest, size_bytes=path.stat().st_size)
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_parquet_rgb_depth_state_roundtrip_and_negative_gain(tmp_path):
    row = make_dataset(tmp_path)
    report = verify_dataset(tmp_path)
    assert report["valid"], report
    table = read_transitions(tmp_path)
    assert table.schema.equals(transition_schema(), check_metadata=True)
    assert table.num_rows == 1
    assert row["coverage_gain"] == pytest.approx(-0.4)
    assert row["reward"] == pytest.approx(-0.4)
    assert row["action_u0"] == 1.25 and row["action_v1"] == 3.5
    np.testing.assert_array_equal(read_rgb(tmp_path, row["after_rgb_path"]), observation(1).rgb)
    support = read_support(tmp_path, row["after_support_path"])
    np.testing.assert_array_equal(support["depth"], observation(1).depth)
    np.testing.assert_array_equal(support["calibration"]["intrinsics"], np.eye(3))
    state = read_support(tmp_path, row["final_state_path"])
    np.testing.assert_array_equal(state["positions"], np.ones((2, 3)))
    assert "simulation-only" in (tmp_path / "DATASET_CARD.md").read_text()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_invalid_is_explicit_and_null_reward(tmp_path, value):
    with DatasetWriter(tmp_path) as writer:
        eid = writer.save_reset(observation(0), seed=0, config={})
        result = replace(
            outcome(observation(0), observation(1)), coverage_after=value, area_after_m2=value
        )
        row = writer.append_outcome(eid, result)
    assert not row["execution_valid"]
    assert row["coverage_after"] is None and row["coverage_gain"] is None
    assert row["reward"] is None and row["terminated"]
    assert read_transitions(tmp_path, valid_only=True).num_rows == 0
    evidence = read_support(tmp_path, row["evidence_path"])
    assert "__nonfinite__" in evidence["raw_metrics"]["coverage_after"]
    assert verify_dataset(tmp_path)["valid"]


def test_finite_invalid_has_no_reward(tmp_path):
    row = make_dataset(tmp_path, valid=False)
    assert row["coverage_gain"] < 0 and row["reward"] is None
    assert verify_dataset(tmp_path)["valid"]


@pytest.mark.parametrize("suffix", [".png", ".npz", ".parquet", ".json"])
def test_hash_tampering_and_missing_artifacts(tmp_path, suffix):
    make_dataset(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    reference = next(p for p in manifest["files"] if p.endswith(suffix))
    path = tmp_path / reference
    path.write_bytes(path.read_bytes() + b"tampered")
    assert not verify_dataset(tmp_path)["valid"]
    path.unlink()
    assert not verify_dataset(tmp_path)["valid"]


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda row: row.update(coverage_gain=1.0), "arithmetic"),
        (lambda row: row.update(coverage_before=0.1), "arithmetic"),
        (lambda row: row.update(step_id=8), "sequential"),
        (lambda row: row.update(transition_id=2), "sequential"),
        (lambda row: row.update(before_observation_id="missing"), "discontinuity"),
        (lambda row: row.update(after_rgb_path="../escape.png"), "relative dataset path"),
        (lambda row: row.update(execution_valid=False), "invalid outcome has trainable reward"),
    ],
)
def test_semantic_tampering_even_with_updated_file_hash(tmp_path, mutate, error):
    make_dataset(tmp_path)
    rewrite_part(tmp_path, mutate)
    report = verify_dataset(tmp_path)
    assert not report["valid"]
    assert any(error in message for message in report["errors"]), report


def test_final_attachments_are_rejected(tmp_path):
    make_dataset(tmp_path, state={"attached_ids": np.array([2, 5])})
    report = verify_dataset(tmp_path)
    assert not report["valid"]
    assert any("attachments" in error for error in report["errors"])


@pytest.mark.parametrize("free,valid", [([True, True], True), ([True, False], False)])
def test_backend_free_mask_without_attached_ids(tmp_path, free, valid):
    make_dataset(
        tmp_path,
        state={
            "free": np.array(free, dtype=bool),
            "positions": np.ones((2, 3)),
            "velocities": np.zeros((2, 3)),
        },
    )
    report = verify_dataset(tmp_path)
    assert report["valid"] is valid, report
    if not valid:
        assert any("free mask" in error for error in report["errors"])


@pytest.mark.parametrize("key", ["positions", "velocities"])
@pytest.mark.parametrize("valid_execution", [True, False])
def test_nonfinite_final_state_only_allowed_on_invalid_execution(tmp_path, key, valid_execution):
    state = {
        "free": np.ones(2, dtype=bool),
        "positions": np.ones((2, 3)),
        "velocities": np.zeros((2, 3)),
    }
    state[key][0, 0] = np.nan
    make_dataset(tmp_path, state=state, valid=valid_execution)
    report = verify_dataset(tmp_path)
    assert report["valid"] is not valid_execution, report
    if valid_execution:
        assert any("nonfinite " + key in error for error in report["errors"])


@pytest.mark.parametrize("override_group", [False, True])
def test_same_generated_seed_family_stays_grouped_under_numeric_perturbation(
    tmp_path, override_group
):
    with DatasetWriter(tmp_path) as writer:
        for index, split in enumerate(("train", "test")):
            eid = writer.save_reset(
                observation(0),
                seed=42,
                config=SceneConfig(),
                cloth_id="same-cloth",
                state={"positions": np.ones((2, 3)) + index * 1e-7},
                reset_kind="drop",
                split=split,
                reset_group_id=f"renamed-{index}" if override_group else None,
            )
            writer.finish_episode(eid, reason="no_valid_pair", terminated=True)
        a, b = writer.manifest["episodes"]
        assert a["reset_content_sha256"] != b["reset_content_sha256"]
        assert a["reset_family_sha256"] == b["reset_family_sha256"]
        if not override_group:
            assert a["reset_group_id"] == b["reset_group_id"]
    report = verify_dataset(tmp_path)
    assert not report["valid"]
    assert any("split leakage" in error for error in report["errors"])


def test_group_split_leakage_including_zero_step_episodes(tmp_path):
    with DatasetWriter(tmp_path) as writer:
        for split in ("train", "test"):
            eid = writer.save_reset(
                observation(0), seed=0, config={}, split=split, reset_group_id="one-reset-family"
            )
            writer.finish_episode(eid, reason="no_valid_pair", terminated=True)
    report = verify_dataset(tmp_path)
    assert not report["valid"]
    assert any("split leakage" in error for error in report["errors"])


@pytest.mark.parametrize("reference", ["../x", "/tmp/x", "a/../x", "./x", "a\\x", "a//../x"])
def test_traversal_rejected(tmp_path, reference):
    with pytest.raises(ValueError):
        relative_path(tmp_path, reference)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        read_rgb(tmp_path, "escape/outside.png")


def test_writer_exclusion_and_closed_writer(tmp_path):
    writer = DatasetWriter(tmp_path)
    with pytest.raises(RuntimeError, match="active writer"):
        DatasetWriter(tmp_path)
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.save_reset(observation(0), seed=0, config={})


def test_interrupted_append_recovery_records_once_and_never_replays(tmp_path, monkeypatch):
    writer = DatasetWriter(tmp_path)
    eid = writer.save_reset(observation(0), seed=3, config={})
    action = outcome(observation(0), observation(1)).action
    writer.begin_attempt(eid, action)
    original_commit = writer._commit
    monkeypatch.setattr(writer, "_commit", lambda: (_ for _ in ()).throw(OSError("disk fault")))
    with pytest.raises(OSError):
        writer.append_outcome(eid, outcome(observation(0), observation(1), final=True))
    monkeypatch.setattr(writer, "_commit", original_commit)
    writer.close()
    with DatasetWriter(tmp_path) as reopened:
        rows = read_transitions(tmp_path).to_pylist()
        assert len(rows) == 1 and rows[0]["reason"] == "interrupted_in_flight"
        assert not rows[0]["execution_valid"] and rows[0]["after_observation_id"] is None
        assert rows[0]["reward"] is None
        assert reopened.manifest["in_flight"] is None
        new_eid = reopened.save_reset(observation(0), seed=4, config={})
        assert new_eid == 1
        reopened.finish_episode(new_eid, reason="no_valid_pair", terminated=True)
    with DatasetWriter(tmp_path):
        assert read_transitions(tmp_path).num_rows == 1
    report = verify_dataset(tmp_path)
    assert report["valid"], report
    assert report["orphan_files"]  # Uncommitted files never become rows during recovery.


def test_interrupt_between_actions_closes_metadata_without_fabricating_action(tmp_path):
    with DatasetWriter(tmp_path) as writer:
        eid = writer.save_reset(observation(0), seed=0, config={})
        writer.append_outcome(eid, outcome(observation(0), observation(1)))
    with DatasetWriter(tmp_path):
        pass
    report = verify_dataset(tmp_path)
    assert report["valid"], report
    assert report["attempts"] == 1


def test_interrupt_after_manifest_replacement_does_not_duplicate_committed_row(
    tmp_path, monkeypatch
):
    writer = DatasetWriter(tmp_path)
    eid = writer.save_reset(observation(0), seed=0, config={})
    result = outcome(observation(0), observation(1), final=True)
    writer.begin_attempt(eid, result.action)
    commit = writer._commit

    def commit_then_fault():
        commit()
        raise OSError("interrupted after atomic replace")

    monkeypatch.setattr(writer, "_commit", commit_then_fault)
    with pytest.raises(OSError):
        writer.append_outcome(eid, result)
    writer.close()
    with DatasetWriter(tmp_path):
        rows = read_transitions(tmp_path).to_pylist()
        assert len(rows) == 1 and rows[0]["execution_valid"]
    report = verify_dataset(tmp_path)
    assert report["valid"], report


def test_nonfinite_action_failure_is_an_attempt_with_null_pixels(tmp_path):
    with DatasetWriter(tmp_path) as writer:
        eid = writer.save_reset(observation(0), seed=0, config={})
        action = GraspAction(((float("nan"), 1), (3, 4)))
        writer.begin_attempt(eid, action)
        row = writer.append_outcome(eid, reason="invalid_pixels")
    assert row["action_u0"] is None and row["action_v0"] == 1
    assert row["reward"] is None and not row["execution_valid"]
    assert verify_dataset(tmp_path)["valid"]


def test_same_id_cannot_silently_replace_image(tmp_path):
    with DatasetWriter(tmp_path) as writer:
        eid = writer.save_reset(observation(0), seed=0, config={})
        changed = replace(observation(0), rgb=observation(1).rgb)
        with pytest.raises(ValueError, match="different content"):
            writer.append_outcome(eid, outcome(changed, observation(2)))


def test_reader_rgb_independent_and_imports_offline(tmp_path):
    row = make_dataset(tmp_path)
    # Block simulator/torch imports even if those packages happen to be installed.
    source = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'genesis', 'torch'} or fullname.startswith('cloth_fling.sim'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from cloth_fling.data import verify_dataset, read_transitions, read_rgb
from cloth_fling.collection import collect_episodes
assert verify_dataset(sys.argv[1])['valid']
assert read_transitions(sys.argv[1]).num_rows == 1
assert read_rgb(sys.argv[1], sys.argv[2]).shape == (5, 7, 3)
"""
    subprocess.run(
        [sys.executable, "-c", source, str(tmp_path), row["after_rgb_path"]],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    # RGB reads need neither the manifest nor supporting arrays.
    (tmp_path / "manifest.json").unlink()
    for support in (tmp_path / "support").iterdir():
        support.unlink()
    assert read_rgb(tmp_path, row["after_rgb_path"]).shape == (5, 7, 3)
