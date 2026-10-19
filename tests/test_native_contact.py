"""Offline impulse/reference and fail-closed native extension checks."""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cloth_fling.contracts import SceneConfig
from cloth_fling.sim.contact import normal_impulse_friction
from cloth_fling.sim.native_contact import validate_native_source


def test_analytic_impulse_static_kinetic_impact_and_exclusions():
    v = np.array([[1.0, 0, 0.2], [0.01, 0, 0.2], [1.0, 0, 0], [1.0, 0, 0]])
    out, j = normal_impulse_friction(
        v,
        np.tile([0, 0, 1.0], (4, 1)),
        np.array([2.0, 2, 0, 2]),
        np.ones(4),
        free=[True, True, True, False],
        static_friction=0.4,
        kinetic_friction=0.3,
    )
    np.testing.assert_allclose(out[:2, 0], [0.4, 0])
    np.testing.assert_array_equal(out[:, 2], v[:, 2])
    np.testing.assert_array_equal(out[2:], v[2:])
    np.testing.assert_allclose(j[0], [-0.6, 0, 0])


def test_random_friction_energy_nonincrease_coulomb_bound_and_normal_invariance():
    rng = np.random.default_rng(31)
    v = rng.normal(size=(1000, 3))
    n = rng.normal(size=v.shape)
    n /= np.linalg.norm(n, axis=1)[:, None]
    m, normal_j = rng.uniform(0.1, 2, (2, len(v)))
    out, j = normal_impulse_friction(
        v, n, normal_j, m, free=np.ones(len(v), bool), static_friction=0.4, kinetic_friction=0.3
    )
    assert np.all(np.sum(out**2, axis=1) <= np.sum(v**2, axis=1) + 1e-12)
    assert np.all(np.linalg.norm(j, axis=1) <= 0.4 * normal_j + 1e-12)
    np.testing.assert_allclose(np.sum(j * n, axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(np.sum(out * n, axis=1), np.sum(v * n, axis=1), atol=1e-14)


def test_source_guard_rejects_version_and_changed_executable_source():
    with pytest.raises(RuntimeError, match="0.4.3"):
        validate_native_source("0.4.4", "")
    with pytest.raises(RuntimeError, match="source"):
        validate_native_source("0.4.3", "class LegacyCoupler: pass")
    # Find source without importing Genesis (which initializes dependencies).
    spec = importlib.util.find_spec("genesis")
    if spec is None:
        pytest.skip("Genesis source unavailable")
    source = (Path(spec.origin).parent / "engine/couplers/legacy_coupler.py").read_text()
    assert len(validate_native_source("0.4.3", source)) == 2
    with pytest.raises(RuntimeError, match="source"):
        validate_native_source("0.4.3", source.replace("energy_loss = 0.0", "energy_loss = 0.1"))
    ast.parse(source)


def test_profile_config_separates_table_from_self_and_preserves_old_json():
    old = SceneConfig().to_dict()
    assert "contact_profile" not in old and "table_static_friction" not in old
    assert SceneConfig(**old) == SceneConfig()
    native = SceneConfig(contact_profile="native_coulomb_v1")
    assert native.table_coefficients == (0.4, 0.3)
    assert native.static_friction == native.kinetic_friction == 0.15
    assert SceneConfig(**json.loads(json.dumps(native.to_dict()))) == native
    for values in (
        {"contact_profile": "bad"},
        {"table_static_friction": -1},
        {"table_kinetic_friction": 1},
    ):
        with pytest.raises(ValueError):
            SceneConfig(**values)


def test_contact_module_import_is_simulator_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cloth_fling.sim.native_contact; "
            "import cloth_fling.contact_diagnostics; "
            'assert "genesis" not in sys.modules; assert "torch" not in sys.modules',
        ],
        check=True,
    )
