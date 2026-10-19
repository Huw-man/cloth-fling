"""Offline installation must not load or initialize Genesis or torch."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_offline_import_and_doctor_do_not_import_simulator():
    code = """
import json, sys
from cloth_fling.cli import doctor_report, load_config
from cloth_fling.contracts import SceneConfig
from cloth_fling.provenance import runtime_provenance
assert load_config(None) == SceneConfig(contact_profile='native_coulomb_v2')
report = doctor_report()
assert runtime_provenance()['torch_cuda_version'] is None
assert not report['simulation_probed']
assert not any(k == 'genesis' or k.startswith('genesis.') for k in sys.modules)
assert 'torch' not in sys.modules
print(json.dumps(report))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["version"] == "0.2.0"


def test_example_configuration_is_portable():
    from cloth_fling.cli import load_config

    config = load_config(str(Path(__file__).parents[1] / "configs" / "cloth50.json"))
    assert config.cloth_size_m == 0.5
    assert config.sim_dt == 0.002


def test_reset_roundtrip_rejects_pickle_and_conflicting_keys(tmp_path):
    from cloth_fling.cli import load_reset

    (tmp_path / "state.json").write_text(json.dumps({"mesh_hash": "mesh", "schema_version": 1}))
    positions = np.zeros((3, 3), dtype=np.float32)
    np.savez_compressed(tmp_path / "state.npz", positions=positions)
    state = load_reset(tmp_path)
    np.testing.assert_array_equal(state["positions"], positions)
    assert state["mesh_hash"] == "mesh"

    np.savez_compressed(tmp_path / "state.npz", mesh_hash=np.array("other"))
    with pytest.raises(ValueError, match="conflicting"):
        load_reset(tmp_path)

    np.savez_compressed(tmp_path / "state.npz", positions=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        load_reset(tmp_path)


def test_cli_records_reset_failure_but_exits_nonzero(monkeypatch, tmp_path, capsys):
    from cloth_fling import cli, collection

    closed = []

    class Backend:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(cli, "_backend", lambda args: Backend())
    monkeypatch.setattr(
        collection,
        "collect_episodes",
        lambda *args, **kwargs: {"reset_failures": 1, "attempts": 0, "stop_reason": "reset_failed"},
    )
    assert cli.main(["collect", "--output", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["reset_failures"] == 1
    assert closed == [True]


def test_historical_dataset_and_replay_inspection_without_simulator(tmp_path):
    from test_collection_pipeline import FakeEnv, execute, select

    from cloth_fling.collection import collect_episodes

    env = FakeEnv()
    snapshot = env.snapshot
    env.snapshot = lambda: {
        **snapshot(),
        "schema_version": 1,
        "config_json": json.dumps(env.config.to_dict()),
    }
    collect_episodes(env, tmp_path, max_actions=1, selector=select, executor=execute)
    code = """
import importlib.abc, sys
class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'genesis', 'torch'}:
            raise AssertionError(f'offline operation imported {fullname}')
sys.meta_path.insert(0, NoSimulator())
from cloth_fling.cli import main
from cloth_fling.data import read_support, read_transitions
root = sys.argv[1]
assert main(['dataset', 'verify', root]) == 0
assert main(['replay', root, '--inspect']) == 0
rows = read_transitions(root).to_pylist()
assert rows[0]['reward'] < 0
assert read_support(root, rows[0]['final_state_path'])['schema_version'] == 1
assert 'genesis' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
