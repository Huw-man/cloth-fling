"""Synthetic offline benchmark evidence, never simulator/heldout outcome inspection."""

import copy
import subprocess
import sys

import numpy as np
import pytest
from test_benchmark_prepare import FixtureEnv, prepare

from cloth_fling.benchmark import (
    MANIFEST,
    SELECTORS,
    BudgetLedger,
    _digest,
    aggregate_episode,
    load_fixture,
    load_qualification,
    record_qualification,
    report_benchmark,
)
from cloth_fling.contracts import ActionOutcome, GraspAction, Observation
from cloth_fling.data import DatasetWriter
from cloth_fling.data.storage import file_sha256, json_bytes, load_json
from cloth_fling.provenance import config_provenance


def qualify(root, manifest):
    ref = root / "diagnostics" / "source.json"
    ref.parent.mkdir()
    ref.write_text('{"synthetic_cpu_evidence": true}')
    item = {
        "fixture_id": manifest["fixtures"][0]["fixture_id"],
        "diagnostic_ref": {"path": "diagnostics/source.json", "sha256": file_sha256(ref)},
    }
    return record_qualification(
        root,
        {
            "manifest_sha256": file_sha256(root / MANIFEST),
            "benchmark_identity": manifest["identity"],
            "passed": True,
            "primitive_config": {"laydown": {"max_strain": 0.1}, "fling": {"max_strain": 0.1}},
            "selection_config": {},
            "laydown_runs": [item],
            "fling_runs": [item],
            "continuity": [item],
        },
    )


def row(index, before, after, valid=True):
    return {
        "step_id": index,
        "transition_id": index,
        "coverage_before": before,
        "coverage_after": after,
        "execution_valid": valid,
        "wall_seconds": 2.0,
        "simulated_seconds": 0.5,
        "reason": "ok" if valid else "strain_guard",
    }


def test_curve_holds_last_confirmed_settled_after_invalid_and_keeps_negative():
    result = aggregate_episode(
        [row(0, 0.5, 0.4), row(1, 0.4, 0.99, False)],
        initial_coverage=0.5,
        complete=True,
        evidence={
            0: {"diagnostics": {"settled": True}},
            1: {"diagnostics": {"settled": False}, "evidence": {"recording": {}}},
        },
    )
    assert result["curve"] == [0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
    assert result["final_improvement"] == pytest.approx(-0.1)
    assert result["observed_invalid_terminal_geometry"][0]["observed_coverage"] == 0.99
    assert result["invalid_attempt_fraction"] == 0.5
    assert result["timing"]["recorded"]["attempts"] == 1
    assert result["timing"]["unrecorded"]["attempts"] == 1
    unconfirmed = aggregate_episode([row(0, 0.5, 0.9)], initial_coverage=0.5, complete=True)
    assert unconfirmed["curve"] == [0.5] * 6
    assert not unconfirmed["complete"]


def write_pair(
    root,
    manifest,
    qualification,
    fixture,
    selector,
    *,
    delta=0.2,
    invalid=False,
    recorded=False,
    claim=True,
):
    ledger = BudgetLedger(root)
    episode_token = (
        ledger.claim_pilot(
            fixture["fixture_id"], selector, qualification_sha256=_digest(qualification)
        )
        if claim
        else None
    )
    source = load_fixture(root, fixture["fixture_id"])
    binding = {
        "manifest_sha256": file_sha256(root / MANIFEST),
        "qualification_sha256": _digest(qualification),
        "fixture_id": fixture["fixture_id"],
        "selector": selector,
    }
    metadata = {
        "episode_metadata": {"benchmark": binding, "initial_settled": True},
        "primitive_config": qualification["primitive_config"]["laydown"],
        "configuration_provenance": config_provenance(FixtureEnv()),
    }
    directory = root / "pilot" / fixture["split"] / fixture["fixture_id"] / selector
    initial = Observation("before", np.zeros((16, 16, 3), dtype=np.uint8))
    terminal = Observation("after", np.ones((16, 16, 3), dtype=np.uint8))
    recording = {}
    if recorded:
        video = directory / "video" / "native.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"synthetic video placeholder, not a native renderer claim")
        recording = {
            "recording": {
                "files": [{"relative_file": "video/native.mp4", "sha256": file_sha256(video)}]
            }
        }
    with DatasetWriter(directory, metadata=metadata) as writer:
        episode = writer.write_episode_reset(
            initial,
            seed=fixture["seed"],
            config=manifest["identity"]["physics_config"],
            state=source,
            reset_source=source,
            split=fixture["split"],
            reset_group_id=fixture["reset_group_id"],
            cloth_id=fixture["cloth_id"],
            reset_id=fixture["fixture_id"],
        )
        policy = "geometry_guided_privileged_depth" if selector == "geometry_guided" else selector
        action = GraspAction(((1, 1), (12, 12)), policy=policy)
        token = (
            ledger.reserve("pilot", wall_seconds=5, metadata={"episode_token": episode_token})
            if claim
            else None
        )
        writer.begin_attempt(episode, action, before=initial)
        c0 = fixture["coverage"]
        outcome = ActionOutcome(
            initial,
            terminal,
            action,
            c0,
            c0 + delta,
            c0 * 0.25,
            (c0 + delta) * 0.25,
            0.25,
            not invalid,
            "strain_guard" if invalid else "ok",
            0.5,
            2.0,
            diagnostics={"settled": not invalid},
            evidence=recording,
        )
        writer.append_outcome(
            episode,
            outcome,
            final_state=source,
            truncated=not invalid,
            end_reason="execution_invalid" if invalid else "max_actions",
        )
        if token:
            ledger.finish(token, wall_seconds=2)
    if episode_token:
        ledger.finish_pilot(episode_token)
    return directory


def test_missing_and_no_pair_denominators_are_not_dropped(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    fixture = next(f for f in manifest["fixtures"] if f["kind"] == "drop")
    write_pair(
        tmp_path, manifest, qualification, fixture, "geometry_guided", delta=-0.2, recorded=True
    )
    report = report_benchmark(tmp_path)
    summary = report["splits"]["development"]["geometry_guided"]
    assert summary["groups"] == 6
    assert summary["observed_episodes"] == 1
    assert summary["mean_final_improvement"] == pytest.approx(-0.2 / 6)
    assert len(report["episodes"]) == 24
    assert sum(e["missing"] for e in report["episodes"]) == 23
    assert not report["heldout_geometry_gate"]["passed"]
    html = (tmp_path / "index.html").read_text()
    assert "<svg" in html and "Negative gain" in html
    assert "video/native.mp4" in html
    assert "geometry_guided" in html
    no_pair = aggregate_episode([], initial_coverage=0.5, end_reason="no_valid_pair", complete=True)
    assert no_pair["curve"] == [0.5] * 6 and no_pair["no_pair"]


def test_paired_heldout_gate_and_one_shot_claims(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    ledger = BudgetLedger(tmp_path)
    heldout = next(f for f in manifest["fixtures"] if f["split"] == "heldout")
    with pytest.raises(ValueError, match="development"):
        ledger.claim_pilot(
            heldout["fixture_id"], "geometry_guided", qualification_sha256=_digest(qualification)
        )
    for fixture in manifest["fixtures"]:
        if fixture["kind"] != "drop":
            continue
        for selector in SELECTORS:
            write_pair(
                tmp_path,
                manifest,
                qualification,
                fixture,
                selector,
                delta=0.45 if selector == "geometry_guided" else 0.1,
            )
    report = report_benchmark(tmp_path)
    assert report["heldout_geometry_gate"]["passed"]
    summary = report["splits"]["heldout"]["geometry_guided"]
    assert summary["success_fraction_coverage_ge_0_95"] == 1
    assert summary["positive_final_improvement_groups"] == 6
    assert report["splits"]["heldout"]["mean_paired_final_difference"] == pytest.approx(0.35)
    with pytest.raises(ValueError, match="already consumed"):
        ledger.claim_pilot(
            heldout["fixture_id"], "geometry_guided", qualification_sha256=_digest(qualification)
        )
    with pytest.raises(ValueError, match="sealed"):
        ledger.reserve("diagnostic_laydown", wall_seconds=1)


def test_invalid_observed_geometry_does_not_pass_gate(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    for fixture in manifest["fixtures"]:
        if fixture["kind"] != "drop":
            continue
        for selector in SELECTORS:
            write_pair(
                tmp_path, manifest, qualification, fixture, selector, delta=0.49, invalid=True
            )
    report = report_benchmark(tmp_path)
    assert not report["heldout_geometry_gate"]["passed"]
    assert report["splits"]["heldout"]["geometry_guided"]["mean_final_improvement"] == 0
    assert report["splits"]["heldout"]["geometry_guided"]["invalid_attempt_fraction"] == 1


def test_qualification_hash_refs_and_heldout_tuning_rejected(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    assert load_qualification(tmp_path) == qualification
    changed = copy.deepcopy(qualification)
    changed["primitive_config"]["laydown"]["max_strain"] = 0.5
    with pytest.raises(FileExistsError):
        record_qualification(tmp_path, changed)
    (tmp_path / "diagnostics" / "source.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact hash"):
        load_qualification(tmp_path)


def test_pilot_rejects_unpaired_reset_config_and_repeated_dataset(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    fixture = next(f for f in manifest["fixtures"] if f["kind"] == "drop")
    directory = write_pair(tmp_path, manifest, qualification, fixture, "geometry_guided")
    dataset = load_json(directory, "manifest.json")
    dataset["runs"][0]["metadata"]["primitive_config"]["max_strain"] = 0.2
    (directory / "manifest.json").write_bytes(json_bytes(dataset))
    with pytest.raises(ValueError, match="primitive configuration"):
        report_benchmark(tmp_path)


def test_missing_qualification_remains_readable_and_cannot_pass(tmp_path):
    prepare(tmp_path)
    report = report_benchmark(tmp_path)
    assert not report["heldout_geometry_gate"]["passed"]
    assert len(report["episodes"]) == 24


def test_offline_import_does_not_load_genesis_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cloth_fling.benchmark; "
            "assert 'genesis' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_pilot_per_episode_five_attempts_even_after_reopen(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    fixture = next(f for f in manifest["fixtures"] if f["kind"] == "drop")
    ledger = BudgetLedger(tmp_path)
    episode = ledger.claim_pilot(
        fixture["fixture_id"], "uniform_valid", qualification_sha256=_digest(qualification)
    )
    for _ in range(5):
        ledger = BudgetLedger(tmp_path)
        token = ledger.reserve("pilot", wall_seconds=1, metadata={"episode_token": episode})
        ledger.finish(token, wall_seconds=0)
    with pytest.raises(RuntimeError, match="episode action limit"):
        ledger.reserve("pilot", wall_seconds=1, metadata={"episode_token": episode})
    ledger.finish_pilot(episode, result={"incomplete": True})
    with pytest.raises(ValueError, match="already consumed"):
        ledger.claim_pilot(
            fixture["fixture_id"], "uniform_valid", qualification_sha256=_digest(qualification)
        )


def test_missing_durable_claim_marks_evidence_incomplete(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    fixture = next(f for f in manifest["fixtures"] if f["kind"] == "drop")
    write_pair(tmp_path, manifest, qualification, fixture, "uniform_valid", claim=False)
    report = report_benchmark(tmp_path)
    observed = next(e for e in report["episodes"] if not e["missing"])
    assert not observed["complete"]
    assert any("missing durable" in error for error in report["integrity_errors"])


@pytest.mark.parametrize("reason", ["settled_coverage_threshold", "initially_unfolded"])
def test_accepted_early_success_end_reasons(tmp_path, reason):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    fixture = next(f for f in manifest["fixtures"] if f["kind"] == "drop")
    directory = write_pair(tmp_path, manifest, qualification, fixture, "geometry_guided")
    dataset = load_json(directory, "manifest.json")
    dataset["episodes"][0]["end_reason"] = reason
    # Dataset permits metadata-only closure after a settled valid outcome.
    dataset["episodes"][0]["closure_source"] = "metadata"
    (directory / "manifest.json").write_bytes(json_bytes(dataset))
    report = report_benchmark(tmp_path)
    observed = next(e for e in report["episodes"] if not e["missing"])
    assert observed["complete"]
    assert observed["end_reason"] == reason


def test_unfinished_dataset_remains_visible_and_cannot_pass(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    fixture = next(f for f in manifest["fixtures"] if f["kind"] == "drop")
    directory = write_pair(tmp_path, manifest, qualification, fixture, "uniform_valid")
    dataset = load_json(directory, "manifest.json")
    episode = dataset["episodes"][0]
    episode.update(status="open", end_reason=None, terminated=False, truncated=False)
    episode.pop("closure_source")
    dataset["runs"][0]["status"] = "open"
    (directory / "manifest.json").write_bytes(json_bytes(dataset))
    report = report_benchmark(tmp_path)
    observed = next(e for e in report["episodes"] if not e["missing"])
    assert not observed["complete"]
    assert report["integrity_errors"]
    assert not report["heldout_geometry_gate"]["passed"]


def test_partial_reset_pool_report_needs_no_qualification_and_has_no_pilot_curves(tmp_path):
    from test_benchmark_prepare import FixtureEnv

    env = FixtureEnv(rejected=range(2005, 2012))
    manifest = prepare(tmp_path, env)
    assert manifest["stop_reason"] == "pool_limitation"
    report = report_benchmark(tmp_path)
    assert report["stages"] == {
        "prepare": "pool_limitation",
        "qualify": "not_evaluated",
        "pilot": "not_evaluated",
        "heldout": "not_evaluated",
    }
    assert report["heldout_geometry_gate"]["passed"] is None
    assert not report["heldout_geometry_gate"]["evaluated"]
    assert report["preparation"]["counts"]["heldout_drop"] == {
        "attempted": 12,
        "accepted": 5,
        "rejected": 7,
        "required": 6,
    }
    for split in ("development", "heldout"):
        for selector in SELECTORS:
            summary = report["splits"][split][selector]
            assert summary["mean_curve"] == []
            assert summary["curve_status"] == "not_evaluated"
            assert summary["mean_final_coverage"] is None
        assert report["splits"][split]["mean_paired_final_difference"] is None
    assert all(e["curve"] == [] and e["missing"] for e in report["episodes"])
    html = (tmp_path / "index.html").read_text()
    assert "Heldout geometry gate: NOT EVALUATED" in html
    assert "Reset preparation" in html and "heldout-drop-2011" in html
    for attempt in report["preparation"]["attempts"]:
        assert attempt["state"]["path"] in html
        assert attempt["state_arrays"]["path"] in html
    assert not (tmp_path / "benchmark-qualification.json").exists()


def test_qualification_diagnostic_source_configuration_mismatch_rejected(tmp_path):
    manifest = prepare(tmp_path)
    qualification = qualify(tmp_path, manifest)
    # Change the diagnostic, then update only its checksum to test source semantics.
    source = tmp_path / "diagnostics" / "source.json"
    provenance = config_provenance(
        FixtureEnv(), reset_source=load_fixture(tmp_path, manifest["fixtures"][0]["fixture_id"])
    )
    provenance["physics_config"] = dict(provenance["physics_config"], sim_dt=0.01)
    source.write_bytes(json_bytes({"provenance": provenance}))
    for category in ("laydown_runs", "fling_runs", "continuity"):
        qualification[category][0]["diagnostic_ref"]["sha256"] = file_sha256(source)
    from cloth_fling.benchmark import _check_qualification

    with pytest.raises(ValueError, match="source configuration mismatch"):
        _check_qualification(tmp_path, qualification, manifest)
