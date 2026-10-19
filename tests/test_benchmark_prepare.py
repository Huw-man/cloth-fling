"""CPU-only fixtures exercise gates without constructing simulator scenes."""

import copy
import json

import numpy as np
import pytest

from cloth_fling.benchmark import (
    BudgetLedger,
    _native_state_hash,
    benchmark_identity,
    load_benchmark,
    load_fixture,
    prepare_benchmark,
)
from cloth_fling.contracts import Observation, SceneConfig


class FixtureEnv:
    config = SceneConfig(contact_profile="native_coulomb_v2", image_size=16)
    runtime_identity = {"engine": "genesis", "engine_version": "1.4.4", "substeps": 4}

    def __init__(self, rejected=(), failed=(), duplicate=False):
        self.rejected = rejected
        self.failed = failed
        self.duplicate = duplicate
        self.calls = []
        self.seed = 0

    def reset(self, seed=0, *, kind="drop"):
        self.seed = seed
        self.kind = kind
        self.calls.append((kind, seed))
        if seed in self.failed:
            raise RuntimeError("reset did not settle")
        return Observation(str(seed), np.zeros((16, 16, 3), dtype=np.uint8))

    def coverage(self):
        value = 0.9 if self.seed in self.rejected else 0.5
        return value, value * 0.25, 0.25

    def snapshot(self):
        rest = np.array([[0.0, 0.0, 0.005], [0.1, 0.0, 0.005], [0.0, 0.1, 0.005]])
        p = rest.copy()
        p[:, 0] += 0.00001 * (self.seed if not self.duplicate else 0)
        state = {
            "schema_version": 2,
            "engine": "genesis",
            "engine_version": "1.4.4",
            "runtime_identity_json": json.dumps(self.runtime_identity),
            "substeps": 4,
            "positions": p,
            "velocities": np.zeros_like(p),
            "free": np.ones(3, bool),
            "rest_positions": rest,
            "edges": np.array([[0, 1], [0, 2], [1, 2]]),
            "faces": np.array([[0, 1, 2]]),
            "mesh_hash": "mesh",
            "config_hash": "config",
            "config_json": json.dumps(self.config.to_dict()),
            "calibration_json": "{}",
            "simulated_time": 1.0,
            "reset_metadata_json": json.dumps(
                {
                    "status": "invalid" if self.seed in self.failed else "valid",
                    "stable_seconds": 0.5,
                    "final_rms_speed_m_s": 0.001,
                    "kind": self.kind,
                    "seed": self.seed,
                }
            ),
        }
        state["state_hash"] = _native_state_hash(state)
        return state


def prepare(tmp_path, env=None):
    return prepare_benchmark(env or FixtureEnv(), tmp_path, pair_finder=lambda *a, **k: [True])


def test_first_six_pre_action_qualification_order_and_identities(tmp_path):
    env = FixtureEnv(rejected=(1000,), failed=(1002,))
    manifest = prepare(tmp_path, env)
    assert manifest["complete"] and manifest["stage_passed"]
    dev = [f for f in manifest["fixtures"] if f["kind"] == "drop" and f["split"] == "development"]
    assert [f["seed"] for f in dev] == [1001, 1003, 1004, 1005, 1006, 1007]
    assert len(manifest["attempts"]) == 17
    failure = next(a for a in manifest["attempts"] if a["seed"] == 1002)
    assert failure["state"] and not failure["accepted"]
    state = load_fixture(tmp_path, dev[0]["fixture_id"])
    assert state["schema_version"] == 2
    assert len(env.calls) == 17
    prepare(tmp_path, env)
    assert len(env.calls) == 17  # Frozen preparation cannot regenerate fixtures.
    assert BudgetLedger(tmp_path).status()["counts"]["prepare_drop"] == 14


def test_pool_limitation_preserves_every_failure_and_stops(tmp_path):
    env = FixtureEnv(rejected=range(1000, 1012))
    manifest = prepare(tmp_path, env)
    assert manifest["stop_reason"] == "pool_limitation"
    assert not manifest["complete"]
    assert len(manifest["attempts"]) == 15
    assert not any(seed >= 2000 for _, seed in env.calls)
    assert len(load_benchmark(tmp_path)["fixtures"]) == 3


def test_fold_failure_freezes_valid_incomplete_manifest(tmp_path):
    manifest = prepare(tmp_path, FixtureEnv(failed=(17,)))
    assert manifest["stop_reason"] == "fixture_limitation"
    assert not manifest["stage_passed"]
    assert len(load_benchmark(tmp_path)["attempts"]) == 3


def test_tamper_safe_npz_and_configuration_rejected(tmp_path):
    manifest = prepare(tmp_path)
    changed = copy.deepcopy(manifest["identity"])
    changed["physics_config"]["sim_dt"] *= 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_benchmark(tmp_path, identity=changed)
    npz = next(tmp_path.glob("fixtures/*.npz"))
    npz.write_bytes(npz.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_benchmark(tmp_path)


def test_manifest_tamper_rejected(tmp_path):
    prepare(tmp_path)
    path = tmp_path / "benchmark-manifest.json"
    path.write_text(path.read_text().replace("development", "heldout"))
    with pytest.raises(ValueError, match="document hash mismatch"):
        load_benchmark(tmp_path)


def test_identical_derivative_state_cannot_cross_splits(tmp_path):
    with pytest.raises(ValueError, match="leakage"):
        prepare(tmp_path, FixtureEnv(duplicate=True))


def test_durable_budget_interruption_and_limits(tmp_path):
    ledger = BudgetLedger(tmp_path)
    token = ledger.reserve("diagnostic_fling", wall_seconds=20, key="uncertain")
    restarted = BudgetLedger(tmp_path)
    assert restarted.status()["charged_wall_seconds"] == 20
    with pytest.raises(RuntimeError, match="unresolved"):
        restarted.reserve("diagnostic_fling", wall_seconds=20)
    restarted.abandon(token)
    with pytest.raises(ValueError, match="already consumed"):
        restarted.reserve("diagnostic_fling", wall_seconds=20, key="uncertain")
    for _ in range(5):
        token = restarted.reserve("diagnostic_fling", wall_seconds=20)
        restarted.finish(token, wall_seconds=2)
    assert restarted.status()["charged_wall_seconds"] == 30
    with pytest.raises(RuntimeError, match="attempt budget"):
        restarted.reserve("diagnostic_fling", wall_seconds=20)
    with pytest.raises(ValueError, match="heldout"):
        restarted.reserve("diagnostic_laydown", wall_seconds=20, metadata={"split": "heldout"})


def test_native_v2_required():
    env = FixtureEnv()
    env.config = SceneConfig()
    with pytest.raises(ValueError, match="native-v2"):
        benchmark_identity(env)
