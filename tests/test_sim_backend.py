"""CPU tests exercise backend state/control rules without importing Genesis."""

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_fling.contracts import SceneConfig
from cloth_fling.sim.backend import GenesisBackend, InvalidResetError, _json, _state_hash
from cloth_fling.sim.coverage import CoverageGrid
from cloth_fling.sim.mesh import mesh_edges, mesh_identity, square_mesh


class Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Cloth:
    def __init__(self, positions):
        self.p = positions.astype(np.float32)
        self.v = np.zeros_like(self.p)
        self.free = np.ones(len(self.p), dtype=bool)
        self.writes = []

    def get_particles_pos(self):
        return Tensor(self.p)

    def get_particles_vel(self):
        return Tensor(self.v)

    def set_particles_pos(self, values, particles_idx_local=None):
        self.writes.append(("position", particles_idx_local))
        self.p[:] = values
        self.v[:] = 0

    def set_particles_vel(self, values, particles_idx_local=None):
        self.writes.append(("velocity", particles_idx_local))
        self.v[slice(None) if particles_idx_local is None else particles_idx_local] = values

    def fix_particles(self, ids, zero_velocity=True):
        self.free[ids] = False
        if zero_velocity:
            self.v[ids] = 0

    def release_particle(self, ids=None):
        self.free[slice(None) if ids is None else ids] = True


@pytest.fixture
def backend():
    b = object.__new__(GenesisBackend)
    b.config = SceneConfig(image_size=16, mesh_spacing_m=0.25, contact_profile="native_coulomb_v2")
    b._runtime_settings = {"array_mode": "ndarray", "deterministic_algorithms": True}
    b._config_json = _json(b.config.to_dict())
    b._config_hash = hashlib.sha256(b._config_json.encode()).hexdigest()
    b.rest_positions, b.faces = square_mesh(0.5, 0.25)
    b.edges = mesh_edges(b.faces)
    b._mesh_hash = mesh_identity(b.rest_positions, b.faces)
    b.cloth = Cloth(b.rest_positions + [0, 0, 0.05])
    b._closed = False
    b._observation = None
    b._render_buffers_ready = False
    b._targets = None
    b.attached_ids = ()
    b.simulated_time = 1.0
    b.reset_metadata = {"kind": "fixture"}
    b.particle_diameter = 0.01
    b._initial_height = 0.006
    b.calibration = {"test": "fixture"}
    b._grid = CoverageGrid()
    b._flat_area = 0.25

    def step():
        for _ in range(b.substeps):
            b.cloth.p += b.cloth.v * (b.config.sim_dt / b.substeps)

    b.scene = SimpleNamespace(
        pbd_solver=SimpleNamespace(get_state=lambda _: SimpleNamespace(free=Tensor(b.cloth.free))),
        sim=SimpleNamespace(cur_substep_local=0),
        step=step,
    )
    b.camera = SimpleNamespace(
        render=lambda **_: (
            np.full((16, 16, 3), 100, dtype=np.uint8),
            np.ones((16, 16), dtype=np.float32),
            None,
            None,
        )
    )
    return b


def test_only_attached_velocity_commands_and_release_momentum(backend):
    b = backend
    b.attached_ids = (0, 2)
    b.cloth.fix_particles([0, 2], zero_velocity=False)
    start = b.get_positions()
    target = start[[0, 2]] + [0, 0, 0.002]
    b.set_attachment_targets(target)
    assert b.cloth.writes == []  # Targets alone never advance physics.
    b.step()
    np.testing.assert_allclose(b.get_positions()[[0, 2]], target, atol=1e-7)
    np.testing.assert_equal(b.get_positions()[[1, 3, 4, 5, 6, 7, 8]], start[[1, 3, 4, 5, 6, 7, 8]])
    assert b.cloth.writes == [("velocity", [0, 2])]
    velocity = b.get_velocities()
    b.release()
    np.testing.assert_array_equal(b.get_velocities(), velocity)
    assert b.attached_ids == () and b._free().all()
    b.step()
    assert len(b.cloth.writes) == 1
    assert b.simulated_time == pytest.approx(1.004)


def test_next_step_target_then_hold(backend):
    b = backend
    b.attached_ids = (0, 2)
    target = b.get_positions()[[0, 2]] + [0, 0, 0.002]
    b.set_attachment_targets(target)
    b.step(3)
    np.testing.assert_allclose(b.get_positions()[[0, 2]], target, atol=1e-7)
    np.testing.assert_allclose(b.get_velocities()[[0, 2]], 0, atol=1e-5)


@pytest.mark.parametrize("value", [np.zeros((1, 3)), np.full((2, 3), np.nan), np.full((2, 3), 5)])
def test_bad_targets_do_not_write(backend, value):
    backend.attached_ids = (0, 2)
    with pytest.raises(ValueError):
        backend.set_attachment_targets(value)
    assert not backend.cloth.writes


@pytest.mark.parametrize("count", [-1, 1.5])
def test_bad_steps(backend, count):
    with pytest.raises(ValueError):
        backend.step(count)


def test_npz_restore_readback_clears_current_attachments_and_time(backend, tmp_path):
    b = backend
    b.cloth.v[0] = [1, 2, 3]
    state = b.snapshot()
    assert state["schema_version"] == 2
    assert state["engine_version"] == "1.4.4"
    assert state["substeps"] == 4
    observation = b.observe()
    path = tmp_path / "state.npz"
    np.savez_compressed(path, **state)
    b.attached_ids = (0, 2)
    b._targets = np.zeros((2, 3))
    b.cloth.fix_particles([0, 2])
    b.cloth.p += 1
    b.simulated_time = 10
    with np.load(path, allow_pickle=False) as stored:
        restored = b.restore(dict(stored))
    np.testing.assert_array_equal(b.get_positions(), state["positions"])
    np.testing.assert_array_equal(b.get_velocities(), state["velocities"])
    assert b.simulated_time == 1
    assert b.attached_ids == () and b._targets is None and b._free().all()
    assert restored.observation_id == observation.observation_id


@pytest.mark.parametrize(
    "corrupt",
    ["content", "shape", "config", "mesh", "attached", "nan", "engine", "runtime", "substeps"],
)
def test_invalid_restore_rejected_before_writes(backend, corrupt):
    b = backend
    state = b.snapshot()
    if corrupt == "content":
        state["positions"][0, 0] += 0.001
    elif corrupt == "shape":
        state["velocities"] = np.zeros((2, 3))
    elif corrupt == "config":
        state["config_hash"] = "other"
    elif corrupt == "mesh":
        state["faces"][0, 0] += 1
    elif corrupt == "attached":
        state["free"][0] = False
    elif corrupt == "nan":
        state["velocities"][0, 0] = np.nan
    elif corrupt == "engine":
        state["engine_version"] = "0.4.3"
    elif corrupt == "runtime":
        state["runtime_identity_json"] = "{}"
    elif corrupt == "substeps":
        state["substeps"] = 8
    if corrupt != "content":
        state["state_hash"] = _state_hash(state)
    with pytest.raises(ValueError):
        b.restore(state)
    assert not b.cloth.writes


def archival_state(backend):
    state = backend.snapshot()
    for key in ("engine", "engine_version", "runtime_identity_json", "substeps"):
        state.pop(key)
    state["schema_version"] = 1
    state["state_hash"] = _state_hash(state)
    return state


def test_old_physical_state_rejected_even_with_identical_mesh(backend):
    state = archival_state(backend)
    with pytest.raises(ValueError, match="schema-1 Genesis 0.4.3"):
        backend.restore(state)
    assert not backend.cloth.writes


@pytest.mark.parametrize("profile", ["legacy_mg_v1", "native_coulomb_v1"])
def test_archival_profiles_parse_but_cannot_construct_runtime(profile):
    config = SceneConfig(contact_profile=profile)
    assert SceneConfig(**config.to_dict()) == config
    with pytest.raises(ValueError, match="archival only.*native_coulomb_v2"):
        GenesisBackend(config)


@pytest.mark.parametrize("version", ["0.4.3", "1.4.3", "1.4.5", "1.4.4.dev0"])
def test_wrong_engine_rejected_before_initialization(monkeypatch, version):
    monkeypatch.setitem(sys.modules, "genesis", SimpleNamespace(__version__=version))
    with pytest.raises(RuntimeError, match="requires Genesis 1.4.4"):
        GenesisBackend(SceneConfig(contact_profile="native_coulomb_v2"))


def test_recorded_geometry_explicitly_latches_out_physics(backend):
    state = archival_state(backend)
    backend.scene.step = lambda: pytest.fail("recorded geometry must never step")
    backend.restore_recorded_geometry(state)
    np.testing.assert_array_equal(backend.get_positions(), state["positions"])
    assert backend.capabilities["render_only"]
    assert not backend.capabilities["physical_restore"]
    assert backend.recorded_geometry_identity["source_engine_version"] == "0.4.3"
    assert backend.simulated_time == state["simulated_time"]
    for operation in (
        lambda: backend.step(),
        lambda: backend.reset(),
        lambda: backend.snapshot(),
        lambda: backend.restore(state),
        lambda: backend.attach(((0, 0), (1, 1))),
        lambda: backend.set_attachment_targets(np.zeros((2, 3))),
    ):
        with pytest.raises(RuntimeError, match="render-only"):
            operation()


@pytest.mark.parametrize("key", ["faces", "rest_positions", "edges", "mesh_hash", "positions"])
def test_recorded_geometry_rejects_conversion_before_mutation(backend, key):
    state = archival_state(backend)
    if key == "mesh_hash":
        state[key] = "different"
    elif key == "positions":
        state[key] = np.zeros((2, 3))
    else:
        state[key].flat[0] += 1
    state["state_hash"] = _state_hash(state)
    with pytest.raises(ValueError):
        backend.restore_recorded_geometry(state)
    assert not backend.cloth.writes
    assert not backend.capabilities["render_only"]


def test_runtime_metadata_is_copy_and_appearance_independent(backend):
    identity = backend.runtime_identity
    assert identity["engine_version"] == "1.4.4"
    identity["native_source_sha256_ast"].clear()
    assert backend.runtime_identity["native_source_sha256_ast"]
    backend.appearance = "unrelated visual appearance"
    assert backend.runtime_identity["substeps"] == 4


def test_observation_array_isolation_and_content_identity(backend):
    b = backend
    one = b.observe()
    one.rgb[:] = 0
    one.depth[:] = 0
    one.calibration.clear()
    two = b.observe()
    assert two.rgb.min() == 100 and two.depth.min() == 1
    assert two.calibration == {"test": "fixture"}
    assert one.observation_id == two.observation_id
    b.cloth.p[0, 0] += 0.01
    b._dirty()
    assert b.observe().observation_id != two.observation_id


def test_renderer_warms_buffers_and_forces_same_step_refresh_without_physics(backend):
    b = backend
    calls = []
    render = b.camera.render

    def capture(**options):
        calls.append(options)
        return render(**options)

    def forbidden_step():
        pytest.fail("render/restore must not step physics")

    b.camera.render = capture
    b.scene.step = forbidden_step
    state = b.snapshot()
    b.observe()
    assert len(calls) == 3  # Warm, dynamic RGB, then calibrated depth-only draw.
    b.observe()
    assert len(calls) == 3  # Cached observation needs no redraw.
    b.restore(state)
    assert len(calls) == 5  # Restore forces RGB and depth refresh without stepping.
    assert [call["rgb"] for call in calls] == [True, True, False, True, False]
    assert [call["depth"] for call in calls] == [False, False, True, False, True]
    assert all(call["force_render"] and not call["antialiasing"] for call in calls)
    assert b.snapshot()["state_hash"] == state["state_hash"]


@pytest.mark.parametrize("failure", ["timeout", "escaped"])
def test_reset_failure_is_explicit_and_clears_attachments(backend, failure):
    b = backend
    b.scene.reset = lambda: None

    def invalid_step():
        b.simulated_time += b.config.sim_dt
        if failure == "escaped":
            b.cloth.p[0, 0] = 0.7
        else:
            b.cloth.v[:] = [0.02, 0, 0]

    b.step = invalid_step
    with pytest.raises(InvalidResetError):
        b.reset(5, kind="flat")
    assert b.reset_metadata["status"] == "invalid"
    assert b.reset_metadata["kind"] == "flat"
    assert b.attached_ids == () and b._free().all()
    if failure == "timeout":
        assert b.simulated_time == pytest.approx(10)
        assert b.reset_metadata["final_rms_speed_m_s"] > 0.01
    else:
        assert b.cloth.p[0, 0] == pytest.approx(0.7)  # No fallback/cleanup reset.
