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
assert load_config(None) == SceneConfig()
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
    assert json.loads(result.stdout)["version"] == "0.1.0"


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
