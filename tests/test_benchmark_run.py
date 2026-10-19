"""Coordinator gates fail closed before expensive or held-out execution."""

import copy
import json
import signal
import time
from dataclasses import asdict

import pytest

from cloth_fling import cli
from cloth_fling.benchmark_run import (
    _attempt,
    _EpisodeGuard,
    _laydown_pass,
    _protected_motion_hash,
    _reservation_deadline,
    _unsafe,
    run_benchmark_stage,
)
from cloth_fling.data.storage import file_sha256
from cloth_fling.sim.primitives import PrimitiveConfig


def result(kind, seed, *, valid=True, final=0.96, gain=0.11):
    return {
        "kind": kind,
        "seed": seed,
        "fixture_id": f"{kind}-{seed}",
        "execution_valid": valid,
        "coverage_after": final,
        "coverage_gain": gain,
    }


def test_laydown_gate_requires_both_named_folds_and_four_distinct_bunches():
    rows = [result("fold", 11), result("fold", 17)]
    rows += [result("drop", i) for i in range(4)]
    assert _laydown_pass(rows)
    assert not _laydown_pass([r for r in rows if r["seed"] != 17])
    assert not _laydown_pass(rows[:2] + [rows[2]] * 4)
    assert not _laydown_pass(rows[:2] + [result("drop", i, valid=False) for i in range(4)])
    assert not _laydown_pass([result("fold", 11, final=0.949), result("fold", 17)] + rows[2:])
    assert not _laydown_pass(rows[:2] + [result("drop", i, gain=0.099) for i in range(4)])


def test_incomplete_preparation_stops_pilot_before_backend_import(tmp_path, monkeypatch):
    import cloth_fling.benchmark as benchmark

    monkeypatch.setattr(
        benchmark,
        "load_benchmark",
        lambda root: {"stage_passed": False, "stop_reason": "pool_limitation"},
    )
    result = run_benchmark_stage("pilot", tmp_path)
    assert result == {
        "stage_passed": False,
        "status": "reset_generation_stop",
        "stop_reason": "pool_limitation",
    }
    assert not (tmp_path / "gpu-lease.json").exists()


def test_failed_qualification_stops_before_scene_construction(tmp_path, monkeypatch):
    import cloth_fling.benchmark as benchmark

    monkeypatch.setattr(benchmark, "load_benchmark", lambda root: {"stage_passed": True})
    monkeypatch.setattr(benchmark, "load_qualification", lambda root: {"passed": False})
    with pytest.raises(ValueError, match="before scene construction"):
        run_benchmark_stage("pilot", tmp_path)
    assert not (tmp_path / "gpu-lease.json").exists()


def test_frozen_preparation_resume_never_constructs_scene(tmp_path, monkeypatch):
    import cloth_fling.benchmark as benchmark

    (tmp_path / "benchmark-manifest.json").write_text("frozen")
    frozen = {"stage_passed": False, "stop_reason": "pool_limitation"}
    monkeypatch.setattr(benchmark, "load_benchmark", lambda root: frozen)
    assert run_benchmark_stage("prepare", tmp_path) is frozen
    assert not (tmp_path / "gpu-lease.json").exists()


def test_revision_cannot_replace_missing_initial_diagnostics_with_new_actions(tmp_path):
    with pytest.raises(ValueError, match="complete initial diagnostic schedule"):
        _attempt(
            None,
            tmp_path,
            None,
            {},
            None,
            {},
            PrimitiveConfig(),
            "missing-fold",
            allow_execute=False,
        )


def test_structured_failure_classification_retains_scientific_failures():
    assert not _unsafe(
        {
            "execution_valid": False,
            "reason": "cloth_strain_limit",
            "diagnostics": {
                "terminal_observation_valid": True,
                "errors": [{"type": "PrimitiveFailure", "message": "cloth_strain_limit"}],
            },
        }
    )
    for reason in (
        "terminal_capture_failed_ValueError",
        "release_cleanup_failed",
        "unexpected_RuntimeError: unknown",
        "execution_exception",
    ):
        assert _unsafe({"execution_valid": False, "reason": reason})
    assert _unsafe({"final_restore_valid": False, "runs": []})
    assert _unsafe({"recordings": [{"recording_failed": True}]})
    assert _unsafe({"infrastructure_errors": ["storage failed"]})


def test_revision_protection_covers_guards_and_non_laydown_motion():
    source = """
class PrimitiveConfig:
    max_strain: float = .10
    laydown_distance_m: float = .30
def execute_action(action):
    if action.primitive == 'laydown':
        lower(.30)
    else:
        swing(.16)
"""
    assert _protected_motion_hash(source) == _protected_motion_hash(
        source.replace("lower(.30)", "lower(.40)")
    )
    assert _protected_motion_hash(source) != _protected_motion_hash(source.replace(".10", ".20"))
    assert _protected_motion_hash(source) != _protected_motion_hash(
        source.replace("swing(.16)", "swing(.25)")
    )


def test_cached_failed_restore_cannot_resume_even_with_valid_child_action(tmp_path):
    directory = tmp_path / "diagnostics" / "fold"
    directory.mkdir(parents=True)
    pair = {"pixels": [[1, 2], [3, 4]]}
    report = {
        "config": asdict(PrimitiveConfig()),
        "fixed_pixels": pair["pixels"],
        "final_restore_valid": False,
        "runs": [{"primitive": "laydown", "execution_valid": True, "reason": "completed"}],
    }
    path = directory / "diagnostic.json"
    path.write_text(json.dumps(report))
    (tmp_path / "benchmark-budget.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "key": "fold",
                        "status": "finished",
                        "metadata": {"fixture_id": "development-fold-11"},
                        "result": {"sha256": file_sha256(path)},
                    }
                ]
            }
        )
    )
    with pytest.raises(RuntimeError, match="cached diagnostic infrastructure stop"):
        _attempt(
            None,
            tmp_path,
            None,
            {"fixture_id": "development-fold-11"},
            None,
            pair,
            PrimitiveConfig(),
            "fold",
        )


def test_reservation_timer_tightens_and_preserves_parent_deadline():
    from cloth_fling.collection import _deadline

    with _deadline(time.monotonic() + 2):
        outer = signal.getitimer(signal.ITIMER_REAL)[0]
        with _reservation_deadline(time.monotonic() + 0.25):
            assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 0.25
        restored = signal.getitimer(signal.ITIMER_REAL)[0]
        assert 1 < restored <= outer
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_guard_rejects_restore_between_actions_and_any_reset():
    class Env:
        def restore(self, state):
            return object()

    guard = _EpisodeGuard(Env(), {"state_hash": "a"})
    guard.actions = 1
    with pytest.raises(RuntimeError, match="inside evolving episode"):
        guard.restore(copy.deepcopy(guard.state))
    with pytest.raises(RuntimeError, match="forbids reset"):
        guard.reset()


def test_cli_exposes_compatible_diagnostics_and_metadata():
    parser = cli.build_parser()
    old = parser.parse_args(["diagnose", "--output", "unused"])
    assert tuple(old.primitives) == ("laydown", "fling")
    new = parser.parse_args(["diagnose", "--output", "unused", "--primitives", "laydown"])
    assert new.primitives == ["laydown"]
    collect = parser.parse_args(
        [
            "collect",
            "--output",
            "unused",
            "--split",
            "heldout",
            "--reset-group-id",
            "drop-2000",
            "--cloth-id",
            "cloth",
            "--settled-coverage-threshold",
            ".95",
        ]
    )
    assert collect.settled_coverage_threshold == 0.95 and collect.split == "heldout"
