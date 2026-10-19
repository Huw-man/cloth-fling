"""Coordinator gates fail closed before expensive or held-out execution."""

import copy

import pytest

from cloth_fling import cli
from cloth_fling.benchmark_run import _EpisodeGuard, _laydown_pass, run_benchmark_stage


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
