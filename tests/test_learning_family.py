import json
from collections import Counter

import pytest
from test_collection_m3 import CoverageEnv, advancing_executor, select

from cloth_fling import learning_family as family
from cloth_fling import learning_pilot as pilot
from cloth_fling.collection import collect_episodes
from cloth_fling.data import read_transitions


def result(policy="initial", score=0.7, success=False, actions=3, failed=False):
    return {
        "policy": policy,
        "score": score,
        "failed": failed,
        "fatal": None,
        "actions": [
            {
                "execution_valid": not failed,
                "coverage_after": score,
                "diagnostics": {
                    "settled": not failed,
                    "final_rms_speed_m_s": 0.001,
                    "final_table_containment": 1.0,
                },
            }
        ]
        * actions,
    }


def test_exact_splits_schedule_and_balanced_repeats():
    cases = family.fixture_cases()
    assert Counter(c["split"] for c in cases) == {"train": 6, "validation": 2, "test": 4}
    families = {}
    for case in cases:
        assert families.setdefault(case["family"], case["split"]) == case["split"]
    assert len({c["id"] for c in cases}) == 12
    schedules = {
        s: family.episode_schedule(s) for s in ("exploration", "refresh", "validation", "test")
    }
    assert {s: len(j) for s, j in schedules.items()} == {
        "exploration": 24,
        "refresh": 24,
        "validation": 8,
        "test": 16,
    }
    assert sum(map(len, schedules.values())) * 3 == family.PROTOCOL["action_limit"] == 216
    for stage in ("validation", "test"):
        jobs = schedules[stage]
        for case in {j["case"] for j in jobs}:
            repeats = [
                [j for j in jobs if j["case"] == case and j["repeat"] == r] for r in range(2)
            ]
            assert [j["policy"] for j in repeats[0]] == [j["policy"] for j in reversed(repeats[1])]
            assert all(pair[0]["seed"] == pair[1]["seed"] for pair in repeats)
    assert len(pilot.fixture_cases()) == 21  # historical default unchanged
    assert pilot.PROTOCOL["coverage_stop"] == 0.95


@pytest.mark.parametrize("coverages", [[0.8], [0.7, 0.8], [0.7, 0.7, 0.8]])
def test_80_percent_stops_without_executing_unused_flings(tmp_path, coverages):
    env = CoverageEnv(initial=0.5)
    report = collect_episodes(
        env,
        tmp_path,
        max_actions=3,
        selector=select,
        executor=advancing_executor(coverages),
        settled_coverage_threshold=0.8,
    )
    rows = read_transitions(tmp_path).to_pylist()
    assert len(rows) == report["valid"] == len(coverages)
    assert rows[-1]["end_reason"] == "settled_coverage_threshold"
    assert rows[-1]["terminated"] and not rows[-1]["truncated"]
    assert len(env.resets) == 1


def test_success_selection_counts_and_ties():
    a = [result(score=0.8)] + [result(score=0.1) for _ in range(3)]
    b = [result("refreshed", score=0.79) for _ in range(4)]
    assert family.choose_checkpoint(a + b)[0] == "initial"
    a = [result() for _ in range(4)]
    b = [result("refreshed", score=0.71) for _ in range(4)]
    assert family.choose_checkpoint(a + b)[0] == "refreshed"
    b = [result("refreshed") for _ in range(4)]
    assert family.choose_checkpoint(a + b)[0] == "initial"
    with pytest.raises(ValueError, match="incomplete"):
        family.choose_checkpoint(a + b[:-1])
    good = [result(score=0.8, actions=1), result(score=0.85, actions=2)]
    metrics = family.policy_metrics(good + [result(score=0.99, failed=True)])
    assert metrics["successes"] == 2
    assert metrics["success_within_actions"] == {"1": 1, "2": 2, "3": 2}
    assert metrics["mean_actions_on_success"] == 1.5
    good[0]["actions"][-1]["diagnostics"]["final_table_containment"] = 0.99
    assert not family.success(good[0])


def prepared(tmp_path):
    journal = family.Journal(tmp_path / "run")
    with journal.operation("prepare") as outputs:
        family.write(journal.root, "fixture-marker.json", {"fixed": True})
        journal.seal(outputs, journal.root / "fixture-marker.json")
    return journal


def test_uncertain_attempt_and_existing_root_cannot_resume(tmp_path):
    journal = prepared(tmp_path)
    job = family.episode_schedule("exploration")[0]
    with pytest.raises(RuntimeError, match="failed intentionally"):
        with journal.operation("exploration"):
            journal.claim_action(job["episode"], 0)
            raise RuntimeError("failed intentionally")
    assert len(list((journal.root / "action-intents").glob("*.json"))) == 1
    with pytest.raises(RuntimeError, match="uncertain"):
        with journal.operation("initial"):
            pytest.fail("resumed")
    with pytest.raises(FileExistsError):
        family.run(journal.root, backend_factory=lambda *a: pytest.fail("backend opened"))
    with pytest.raises(RuntimeError, match="already claimed"):
        journal.stage = "exploration"
        journal.claim_action(job["episode"], 0)


def test_durable_intent_survives_ledger_gap(tmp_path, monkeypatch):
    journal = prepared(tmp_path)
    job = family.episode_schedule("exploration")[0]
    journal.stage = "exploration"
    monkeypatch.setattr(journal, "save", lambda: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        journal.claim_action(job["episode"], 0)
    intent = json.loads(next((journal.root / "action-intents").glob("*.json")).read_text())
    assert intent["episode"] == job["episode"]
    assert family.read(journal.root, "ledger.json")["actions"] == []


def test_reservation_and_caps_precede_claims(tmp_path):
    journal = prepared(tmp_path)
    journal.stage = "refresh"
    job = family.episode_schedule("refresh")[0]
    journal.started -= 21600 - 7200 - 179
    with pytest.raises(TimeoutError):
        journal.claim_action(job["episode"], 0)
    assert not journal.document["actions"]
    journal.started += 21600 - 7200 - 179
    journal.document["actions"] = [{}] * 216
    with pytest.raises(RuntimeError, match="action limit"):
        journal.claim_action(job["episode"], 0)
    journal.stage = "test"
    assert journal.reserve == 0


def test_sealed_outputs_rechecked_before_dependent_stage(tmp_path):
    journal = prepared(tmp_path)
    family.write(journal.root, "fixture-marker.json", {"fixed": False})
    with pytest.raises(ValueError, match="sealed artifact"):
        with journal.operation("exploration"):
            pytest.fail("executed after corruption")


def test_training_cannot_consume_validation_rows(tmp_path, monkeypatch):
    journal = prepared(tmp_path)
    monkeypatch.setattr(family, "fixtures", lambda root: ([], {}))
    jobs = family.episode_schedule("exploration")
    rows = [{"episode": j["episode"], "case": j["case"], "split": "validation"} for j in jobs]
    family.write(journal.root, "exploration-results.json", {"episodes": rows})
    with pytest.raises(ValueError, match="non-training"):
        family.training_datasets(journal, "initial")


def test_run_routes_80_percent_and_containment(tmp_path, monkeypatch):
    import cloth_fling.learning as learning

    journal = prepared(tmp_path)
    cases = family.fixture_cases()
    monkeypatch.setattr(family, "fixtures", lambda root: (cases, {c["id"]: c["id"] for c in cases}))
    monkeypatch.setattr(learning, "predictor", lambda p: pytest.fail("random round opened model"))
    observed = []

    def episode(env, journal, case, state, name, **kwargs):
        observed.append(kwargs)
        assert kwargs["coverage_stop"] == 0.8 and kwargs["require_containment"]
        assert kwargs["reserve_seconds"] == 7200
        assert state == case["id"]
        folder = journal.root / "episodes" / name
        family.write(folder, "dataset/manifest.json", {})
        return result()

    monkeypatch.setattr(pilot, "episode", episode)
    with journal.operation("exploration") as outputs:
        family.run_episodes(None, journal, "exploration", outputs)
    assert len(observed) == 24
    assert len(journal.document["episodes"]) == 24


def test_report_preserves_unfinalized_dataset_accounting(tmp_path):
    journal = prepared(tmp_path)
    job = family.episode_schedule("exploration")[0]
    journal.stage = "exploration"
    journal.claim_action(job["episode"], 0)
    folder = journal.root / "episodes" / job["episode"]
    family.claim(folder / "intent.json", job)
    collect_episodes(
        CoverageEnv(),
        folder / "dataset",
        max_actions=1,
        selector=select,
        executor=advancing_executor([0.81]),
        settled_coverage_threshold=0.8,
    )
    family.write(journal.root, "scene-cleanup.json", {"passed": True})
    report = family.report(journal.root)
    assert report["claimed_actions"] == report["dataset_rows"] == 1
    assert not report["pipeline_complete"]
    assert report["episodes"][0]["finalized"] is False
    assert any("uncertain episode" in error for error in report["integrity_errors"])
    assert not any("intent/row" in error for error in report["integrity_errors"])
