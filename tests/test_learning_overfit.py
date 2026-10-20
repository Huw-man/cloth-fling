"""Offline proofs for the bounded single-reset campaign, not behavior qualification."""

import copy
import subprocess
import sys

import numpy as np
import pytest
from test_collection_m3 import CoverageEnv, advancing_executor
from test_collection_pipeline import select

from cloth_fling import learning_overfit as overfit
from cloth_fling.collection import collect_episodes
from cloth_fling.data.storage import file_sha256
from cloth_fling.reference_run import save_bundle


def successful_result():
    return {
        "failed": False,
        "fatal": None,
        "actions": [
            {
                "execution_valid": True,
                "coverage_after": 0.98,
                "diagnostics": {
                    "settled": True,
                    "final_table_containment": 1.0,
                    "final_rms_speed_m_s": 0.01,
                    "terminal_observation_valid": True,
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("coverage_after", 0.9799),
        ("execution_valid", False),
        ("settled", False),
        ("final_table_containment", 0.9999),
        ("final_rms_speed_m_s", 0.010001),
        ("terminal_observation_valid", False),
    ],
)
def test_success_requires_complete_settled_contained_result(field, value):
    result = successful_result()
    assert overfit.episode_succeeded(result)
    action = result["actions"][-1]
    (action if field in action else action["diagnostics"])[field] = value
    assert not overfit.episode_succeeded(result)


def test_overfit_containment_gate_continues_then_stops(tmp_path):
    from cloth_fling.contracts import scene_config_from_dict
    from cloth_fling.frozen_laydown import frozen_laydown_spec

    env = CoverageEnv(initial=0.5)
    env.config = scene_config_from_dict(frozen_laydown_spec()["scene"])
    xmin, xmax, ymin, ymax = env.config.table_bounds_m
    env.get_positions = lambda: np.array(
        [[xmax + 0.01 if env.index == 1 else (xmin + xmax) / 2, (ymin + ymax) / 2, 0.3]]
    )
    result = collect_episodes(
        env,
        tmp_path,
        max_actions=3,
        selector=select,
        executor=advancing_executor([0.99, 0.98, 0.5]),
        settled_coverage_threshold=0.98,
        success_requires_containment=True,
    )
    assert result["attempts"] == 2
    assert result["episode_reports"][0]["end_reason"] == "settled_coverage_threshold"
    assert len(env.resets) == 1


def test_fixture_copy_preserves_bytes_and_rejects_changed_identity(tmp_path, monkeypatch):
    source = tmp_path / "source"
    folder = source / "fixture"
    folder.mkdir(parents=True)
    state = {
        "state_hash": "fixed",
        "free": np.ones(3, bool),
        "positions": np.arange(9).reshape(3, 3),
    }
    ref = save_bundle(folder, "fixture", state)
    monkeypatch.setattr(overfit, "FIXTURE_SHA256", ref["sha256"])
    monkeypatch.setattr(overfit, "STATE_HASH", "fixed")
    case = {
        "id": overfit.FIXTURE_ID,
        "state_hash": "fixed",
        "fixture": {"path": "fixture/fixture.json", "sha256": ref["sha256"]},
    }
    overfit.write(source, "fixtures-manifest.json", {"cases": [case]})
    journal = overfit.Journal(tmp_path / "run")
    _, copied = overfit.prepare_fixture(journal, source)
    np.testing.assert_array_equal(copied["positions"], state["positions"])
    for name in ("fixture.json", "fixture.npz"):
        assert (folder / name).read_bytes() == (journal.root / "fixture" / name).read_bytes()
    (folder / "fixture.npz").write_bytes(b"corrupt")
    other = overfit.Journal(tmp_path / "other")
    with pytest.raises(ValueError):
        overfit.prepare_fixture(other, source)


def exercise_schedule(tmp_path, success_at=None, confirmation_failure=False):
    journal = overfit.Journal(tmp_path / "run")
    fits, epsilons = [], []

    def fit(journal, datasets, checkpoint, previous):
        assert all("check" not in str(p) and "confirmation" not in str(p) for p in datasets)
        fits.append((len(datasets), previous))
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(b"weights")
        overfit.write(checkpoint.parent, checkpoint.with_suffix(".json").name, {})

    def episode(env, journal, case, fixture, name, *, predict, epsilon):
        epsilons.append((journal.kind, epsilon))
        folder = journal.root / "episodes" / name
        overfit.claim(folder / "intent.json", {})
        for step in range(3):
            journal.claim_action(name, step)
        passed = journal.stage == success_at or journal.kind == "confirmation"
        if confirmation_failure and journal.kind == "confirmation":
            passed = False
        result = successful_result()
        result.update(
            episode=name, kind=journal.kind, success=passed, score=0.98 if passed else 0.6
        )
        result["actions"] *= 3
        overfit.write(folder, "result.json", result)
        overfit.write(folder, "cleanup.json", {"passed": True})
        overfit.write(
            folder, "restoration.json", {"passed": True, "state_hash": overfit.STATE_HASH}
        )
        journal.document["episodes"].append(name)
        journal.save()
        return result

    result = overfit.develop(
        None,
        journal,
        {},
        {},
        fit_fn=fit,
        episode_fn=episode,
        make_predictor=lambda checkpoint: object(),
    )
    return result, journal, fits, epsilons


def test_full_schedule_and_reserved_confirmations_exactly_300(tmp_path):
    result, journal, fits, epsilons = exercise_schedule(tmp_path, success_at="check-07")
    assert result == "qualified"
    assert len(journal.document["actions"]) == 300
    assert [n for n, _ in fits] == list(range(20, 91, 10))
    assert fits[0][1] is None and all(previous for _, previous in fits[1:])
    assert all(epsilon == 0 for kind, epsilon in epsilons if kind in ("check", "confirmation"))
    assert len(list((journal.root / "checkpoints").glob("*.pt"))) == 8


def test_first_success_freezes_and_confirmation_failure_does_not_tune(tmp_path):
    result, journal, fits, _ = exercise_schedule(
        tmp_path, success_at="check-00", confirmation_failure=True
    )
    assert result == "confirmation_failed" and len(fits) == 1
    assert len(journal.document["actions"]) == 66
    assert not (journal.root / "stage-intents" / "refresh-01.json").exists()
    assert not (journal.root / "stage-intents" / "confirmation-01.json").exists()


def test_no_success_does_not_spend_confirmation_reserve(tmp_path):
    result, journal, fits, _ = exercise_schedule(tmp_path)
    assert result == "not_qualified" and len(fits) == 8
    assert len(journal.document["actions"]) == 294
    assert not (journal.root / "selection.json").exists()


def test_time_and_action_budgets_reject_before_claim(tmp_path):
    journal = overfit.Journal(tmp_path / "run")
    journal.started -= overfit.WALL_SECONDS - overfit.FINAL_RESERVE_SECONDS - 100
    with pytest.raises(TimeoutError):
        journal.claim_action("episode", 0)
    assert not journal.document["actions"]
    journal.kind = "confirmation"
    journal.claim_action("confirmation", 0)
    with pytest.raises(RuntimeError, match="already claimed"):
        journal.claim_action("confirmation", 0)
    journal.document["actions"] = [{}] * 300
    with pytest.raises(RuntimeError, match="budget"):
        journal.claim_action("confirmation", 1)


def test_uncertain_attempt_accounted_and_existing_run_never_opens_scene(tmp_path):
    journal = overfit.Journal(tmp_path / "run")
    overfit.claim(journal.root / "action-intents" / "0000.json", {"orphan": True})
    before = file_sha256(journal.root / "ledger.json")
    result = overfit.report(journal.root, verify=False)
    assert result["claimed_actions"] == 1 and not result["qualified"]
    assert "action intent/ledger mismatch" in result["integrity_errors"]
    with pytest.raises(FileExistsError):
        overfit.run(journal.root, backend_factory=lambda _: pytest.fail("opened scene"))
    assert file_sha256(journal.root / "ledger.json") == before


def test_checks_and_unrelated_data_cannot_enter_fit(tmp_path):
    folder = tmp_path / "check" / "dataset"
    folder.mkdir(parents=True)
    overfit.write(folder.parent, "result.json", {"kind": "check", "case": overfit.FIXTURE_ID})
    with pytest.raises(ValueError, match="cannot enter"):
        overfit.fit_checkpoint(None, [folder], tmp_path / "model.pt", None)


def test_restore_exact_checks_velocity_not_just_positions():
    from test_reference_pilot import Env

    env = Env()
    state = env.snapshot()
    overfit.pilot.restore_exact(env, state)
    changed = copy.deepcopy(state)
    changed["velocities"][0, 0] += 0.1
    with pytest.raises(RuntimeError, match="restoration"):
        overfit.pilot.restore_exact(env, changed)


def test_offline_import_and_cli_do_not_import_simulator_or_torch():
    script = (
        "import sys; import cloth_fling.learning_overfit; "
        "from cloth_fling.cli import build_parser; "
        "args=build_parser().parse_args(['learning-overfit','report']); "
        "assert args.command=='learning-overfit'; "
        "assert 'torch' not in sys.modules and 'genesis' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
