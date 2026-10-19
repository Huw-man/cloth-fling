"""Offline installation must not load or initialize Genesis or torch."""

import json
import subprocess
import sys
from pathlib import Path


def test_offline_import_and_doctor_do_not_import_simulator():
    code = """
import json, sys
from cloth_fling.cli import doctor_report, load_config
from cloth_fling.contracts import SceneConfig
assert load_config(None) == SceneConfig()
report = doctor_report()
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
