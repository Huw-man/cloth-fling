import json
import time

import numpy as np
import pytest
from test_reference_pilot import ACTION, Env

from cloth_fling import learning_pilot as pilot
from cloth_fling.data import read_support, read_transitions


def test_split_families_and_schedule_are_bounded():
    cases = pilot.fixture_cases()
    assert len(cases) == 21
    families = {}
    for case in cases:
        assert families.setdefault(case["family"], case["split"]) == case["split"]
    assert sum(c["split"] == "train" for c in cases) == 9
    assert 9 * 8 * 3 + 9 * 5 * 3 + 3 * 3 * 3 + 9 * 2 * 3 == pilot.ACTION_LIMIT


def test_fold_preserves_rest_and_centers_footprint():
    from cloth_fling.contracts import scene_config_from_dict
    from cloth_fling.frozen_laydown import frozen_laydown_spec
    from cloth_fling.reference_geometry import reference_mesh

    config = scene_config_from_dict(frozen_laydown_spec()["scene"])
    rest, _, _ = reference_mesh(config)
    original = rest.copy()
    for case in pilot.fixture_cases():
        result = pilot.folded_positions(rest, config, case["fraction"], case["angle_deg"])
        np.testing.assert_allclose(
            (result[:, :2].max(0) + result[:, :2].min(0)) / 2, [0.55, 0], atol=1e-9
        )
        assert np.isfinite(result).all() and result[:, 2].min() >= 0.3
    np.testing.assert_array_equal(rest, original)


def ready_journal(tmp_path):
    root = tmp_path / "pilot"
    pilot.initialize(root)
    journal = pilot.Journal(root)
    with journal.operation("prepare"):
        pilot.write(root, "fixtures-manifest.json", {"test": True})
    return root, journal


def test_uncertain_stage_and_action_never_replay(tmp_path):
    root, journal = ready_journal(tmp_path)
    with pytest.raises(ZeroDivisionError):
        with journal.operation("exploration"):
            journal.claim_action("episode", 0)
            1 / 0
    again = pilot.Journal(root)
    assert len(again.document["actions"]) == 1
    with pytest.raises(RuntimeError, match="terminal or uncertain"):
        with again.operation("exploration"):
            pytest.fail("resumed physical work")
    assert pilot.report(root)["status"] == "incomplete"
    with pytest.raises(FileExistsError):
        pilot.initialize(root)


def test_budget_reserves_final_stages_and_counts_claims(tmp_path):
    _, journal = ready_journal(tmp_path)
    journal.stage = "exploration"
    journal.document["charged_seconds"] = pilot.LIMIT_SECONDS - pilot.COLLECTION_RESERVE_SECONDS
    with pytest.raises(TimeoutError):
        journal.claim_action("e", 0)
    assert not journal.document["actions"]
    journal.document["charged_seconds"] = 0
    journal.document["actions"] = [{}] * pilot.ACTION_LIMIT
    with pytest.raises(RuntimeError, match="action budget"):
        journal.claim_action("e", 0)


def test_stage_order_and_sealed_outputs(tmp_path):
    root, journal = ready_journal(tmp_path)
    with pytest.raises(RuntimeError, match="prerequisite"):
        with journal.operation("initial"):
            pass
    pilot.write(root, "fixtures-manifest.json", {"altered": True})
    with pytest.raises(ValueError, match="artifact changed"):
        pilot.Journal(root)
    assert not pilot.report(root)["pipeline_complete"]


def test_three_action_chain_and_signed_labels(tmp_path, monkeypatch):
    import cloth_fling.reference_pilot as old

    root, journal = ready_journal(tmp_path)
    env = Env()
    env.edges = np.array([[0, 1]])
    env.rest_edge_lengths = np.linalg.norm(
        env.rest_positions[[1]] - env.rest_positions[[0]], axis=1
    )
    monkeypatch.setattr(old, "visible_particle", lambda pixel, *args: 20 if pixel[0] else 0)

    def select(self, observation, config, **kwargs):
        self.selection = {
            "encoding": {"codec": "test"},
            "observation_id": observation.observation_id,
            "valid_action_mask": np.ones((96, 64, 64), bool),
        }
        return ACTION

    monkeypatch.setattr(pilot.EpisodePolicy, "__call__", select)
    case = {"id": "fake", "family": "family", "split": "train"}
    with journal.operation("exploration"):
        result = pilot.episode(env, journal, case, env.snapshot(), "fake")
        pilot.write(root, "exploration-results.json", {"episodes": [result]})
    assert len(result["actions"]) == 3 and env.restore_calls == 1
    assert all(d["coverage_gain"] < 0 for d in result["actions"])
    assert len(journal.document["actions"]) == 3
    dataset = root / "episodes" / "fake" / "dataset"
    rows = read_transitions(dataset).to_pylist()
    assert rows[1]["before_rgb_path"] == rows[0]["after_rgb_path"]
    support = read_support(dataset, rows[1]["evidence_path"])
    assert support["evidence"]["selection"]["encoding"]["codec"] == "test"


def test_pairing_counts_failures_and_weights_families():
    results = []
    for family in ("a", "b", "c"):
        for variant in range(3):
            for policy, score in (("uniform", 0.4), ("initial", 0.5)):
                results.append(
                    {
                        "family": family,
                        "case": f"{family}{variant}",
                        "policy": policy,
                        "score": score,
                        "failed": False,
                    }
                )
    assert pilot.compare(results)["behavior_positive"]
    results[-1].update(score=0, failed=True)
    assert not pilot.compare(results)["behavior_positive"]
    with pytest.raises(ValueError, match="incomplete"):
        pilot.compare(results[:-1])


def test_offline_report_retains_uncertain_claim(tmp_path):
    root, journal = ready_journal(tmp_path)
    ledger = journal.document
    ledger["status"] = "running_never_resume"
    ledger["stages"].append(
        {"stage": "exploration", "status": "claimed_never_retry", "started_unix": time.time() - 1}
    )
    pilot.write(root, "ledger.json", ledger)
    result = pilot.report(root)
    assert result["status"] == "incomplete_uncertain_never_resume"
    assert not result["behavior_positive"]
    assert json.loads((root / "ledger.json").read_text())["status"] == "running_never_resume"


def test_prediction_errors_are_fatal_without_random_fallback(monkeypatch):
    from types import SimpleNamespace

    batch = SimpleNamespace(
        rgb=np.zeros((96, 64, 64, 3), np.uint8), feasible=np.ones((96, 64, 64), bool)
    )
    monkeypatch.setattr(pilot, "build_action_batch", lambda *args: batch)
    policy = pilot.EpisodePolicy(lambda _: np.full((96, 64, 64), np.nan), epsilon=0)
    with pytest.raises(ValueError, match="finite"):
        policy(SimpleNamespace(), None, rng=np.random.default_rng(0))
    assert policy.fatal and policy.selection is None


def test_unsettled_episode_stops_without_replacement(tmp_path, monkeypatch):
    import cloth_fling.reference_pilot as old

    root, journal = ready_journal(tmp_path)
    env = Env()
    env.edges = np.array([[0, 1]])
    env.rest_edge_lengths = np.linalg.norm(
        env.rest_positions[[1]] - env.rest_positions[[0]], axis=1
    )
    monkeypatch.setattr(old, "visible_particle", lambda pixel, *args: 20 if pixel[0] else 0)
    execute = old.execute_frozen_laydown

    def unsettled(*args, **kwargs):
        outcome = execute(*args, **kwargs)
        outcome.diagnostics["settled"] = False
        return outcome

    def select(self, observation, config, **kwargs):
        self.selection = {
            "encoding": {"codec": "test"},
            "observation_id": observation.observation_id,
            "valid_action_mask": np.ones((96, 64, 64), bool),
        }
        return ACTION

    monkeypatch.setattr(old, "execute_frozen_laydown", unsettled)
    monkeypatch.setattr(pilot.EpisodePolicy, "__call__", select)
    case = {"id": "fake", "family": "family", "split": "train"}
    with journal.operation("exploration"):
        result = pilot.episode(env, journal, case, env.snapshot(), "unsettled")
        pilot.write(root, "exploration-results.json", {"episodes": [result]})
    assert result["failed"] and result["score"] == 0
    assert len(result["actions"]) == len(journal.document["actions"]) == 1
    assert env.restore_calls == 1


def test_crash_after_intent_before_ledger_save_is_accounted_and_not_resumed(tmp_path):
    root, journal = ready_journal(tmp_path)
    pilot.claim(root / "action-intents" / "0000.json", {"status": "claimed_never_retry"})
    with pytest.raises(RuntimeError, match="intent/ledger gap"):
        pilot.Journal(root)
    result = pilot.report(root)
    assert result["claimed_actions"] == 1
    assert result["status"] == "incomplete_artifact_regression"


def test_stage_has_outer_wall_deadline_before_backend_work(tmp_path, monkeypatch):
    from contextlib import contextmanager

    root = tmp_path / "pilot"
    pilot.initialize(root)
    deadlines = []

    @contextmanager
    def deadline(value):
        deadlines.append(value)
        if len(deadlines) == 1:
            raise TimeoutError("outer budget")
        yield

    monkeypatch.setattr(pilot, "_reservation_deadline", deadline)
    monkeypatch.setattr(
        pilot, "reference_lease", lambda root: __import__("contextlib").nullcontext()
    )
    with pytest.raises(TimeoutError, match="outer budget"):
        pilot.execute_stage(
            root, "prepare", backend_factory=lambda config: pytest.fail("opened scene")
        )
    assert pilot.read(root, "ledger.json")["status"] == "incomplete"


def test_failed_report_does_not_retry_expensive_verification(tmp_path, monkeypatch):
    root, journal = ready_journal(tmp_path)
    directory = root / "episodes" / "test" / "dataset"
    directory.mkdir(parents=True)
    pilot.write(directory, "manifest.json", {})
    pilot.write(
        directory.parent,
        "result.json",
        {
            "episode": "test",
            "failed": True,
            "actions": [],
            "score": 0,
            "dataset_manifest_sha256": pilot.file_sha256(directory / "manifest.json"),
            "collection": {"episode_reports": [{"initial_coverage": 0.5}]},
        },
    )
    monkeypatch.setattr(pilot, "verify_dataset", lambda _: pytest.fail("retried verification"))
    result = pilot.report(root, verify=False)
    assert not result["pipeline_complete"] and not result["verification_completed"]
    assert not result["behavior_positive"]
    with pytest.raises(pytest.fail.Exception, match="retried verification"):
        pilot.report(root)


def test_refused_terminal_command_preserves_successful_report(tmp_path, monkeypatch):
    root, journal = ready_journal(tmp_path)
    journal.document["status"] = "complete"
    journal.save()
    pilot.write(root, "pilot-report.json", {"status": "complete", "pipeline_complete": True})
    original = (root / "pilot-report.json").read_bytes()

    def reject(*args, **kwargs):
        raise RuntimeError("terminal pilot cannot execute")

    monkeypatch.setattr(pilot, "execute_stage", reject)
    assert pilot.main(["collect", "--stage", "refresh", "--output", str(root)]) == 1
    assert (root / "pilot-report.json").read_bytes() == original
    assert not (root / "command-failure.json").exists()
