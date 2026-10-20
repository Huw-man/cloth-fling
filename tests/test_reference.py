"""Reference contracts and learning interfaces. These are not physical qualification."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from cloth_fling.contracts import (
    Observation,
    ReferenceSceneConfig,
    SceneConfig,
    scene_config_from_dict,
)
from cloth_fling.reference_geometry import material_line, reference_mesh, sample_line, triangle_area
from cloth_fling.reference_motion import ReferenceMotionConfig, execute_reference_action
from cloth_fling.reference_run import ReferenceBudget, placement_passed
from cloth_fling.sim.coverage import CoverageGrid
from cloth_fling.sim.mesh import mesh_edges, square_mesh
from cloth_fling.sim.reference import split_contact
from cloth_fling.spatial_actions import build_action_batch, choose_action


def test_reference_config_identity_roundtrip_and_historical_defaults():
    config = ReferenceSceneConfig()
    assert scene_config_from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert config.areal_density_kg_m2 * 0.24 == pytest.approx(config.cloth_mass_kg)
    assert config.table_bounds_m == pytest.approx((0.15, 0.95, -0.5, 0.5))
    assert "profile" not in SceneConfig().to_dict()
    for changes in (
        {"cloth_mass_kg": 0.2},
        {"substeps": 4},
        {"controller_dt": 0.0205},
        {"contact_profile": "native_coulomb_v2"},
        {"wrinkle_amplitude_m": np.nan},
    ):
        with pytest.raises(ValueError):
            replace(config, **changes)
    with pytest.raises(ValueError):
        ReferenceMotionConfig(max_simulated_seconds=36)


def test_reference_mesh_area_relief_and_native_material_line():
    config = ReferenceSceneConfig()
    p, f, scale = reference_mesh(config)
    assert p.shape == (61 * 41, 3)
    assert f.shape == (4800, 3)
    assert 0 < scale < 1
    assert triangle_area(p, f) == pytest.approx(0.24, abs=1e-12)
    assert np.abs(p[:, 2]).max() == pytest.approx(0.005)
    np.testing.assert_array_equal(reference_mesh(config)[0], p)
    a, b = p[f[0]].mean(axis=0), p[f[-1]].mean(axis=0)
    indices, weights = material_line(p, f, a, b)
    assert indices.shape == (65, 3)
    line = sample_line(p, indices, weights)
    np.testing.assert_allclose(line[[0, -1]], [a, b], atol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    np.testing.assert_allclose(sample_line(p + [1, 2, 3], indices, weights), line + [1, 2, 3])


def test_rectangular_world_coverage_negative_and_unclipped():
    p, f = square_mesh(0.2, 0.05)
    p += [0.55, 0, 0.3]
    grid = CoverageGrid(bounds=(0.15, 0.95, -0.3, 0.3), cell_size=0.002)
    assert grid.mask(p, f).shape == (300, 400)
    reference = grid.area(p, f)
    assert reference == pytest.approx(0.04)
    larger = p.copy()
    larger[:, :2] = (larger[:, :2] - [0.55, 0]) * 1.2 + [0.55, 0]
    assert grid.area(larger, f) / reference == pytest.approx(1.44)
    assert grid.area(p + [2, 0, 0], f) - reference < 0


def test_split_contact_exclusions_normal_preservation_and_dissipation():
    p = np.array([[0.5, 0, 0.305]] * 6)
    p[2, 2] += 0.01
    p[3, 0] = 1.0
    p[4, 2] += 2e-6
    before = np.zeros_like(p)
    before[:, 2] = -0.1
    after = np.zeros_like(p)
    after[:, 0] = [0.02, 0.2, 0.2, 0.2, 0.2, 0]
    free = np.ones(6, dtype=bool)
    free[1] = False
    corrected, ids = split_contact(
        p, before, after, free=free, bounds=(0.15, 0.95, -0.5, 0.5), support_z=0.305, dt=0.001
    )
    np.testing.assert_array_equal(ids, [0])
    assert corrected[0, 0] == 0  # No reversal even when impulse exceeds slip.
    np.testing.assert_array_equal(corrected[1:], after[1:])
    np.testing.assert_array_equal(corrected[:, 2], after[:, 2])
    assert (np.sum(corrected**2, axis=1) <= np.sum(after**2, axis=1)).all()
    # No native normal response: gravity alone must not invent support impulse.
    after[:, 2] = before[:, 2] - 9.81 * 0.001
    corrected, ids = split_contact(
        p, before, after, free=free, bounds=(0.15, 0.95, -0.5, 0.5), support_z=0.305, dt=0.001
    )
    np.testing.assert_allclose(corrected, after, atol=1e-15)


def observation():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[25:103, 20:108] = [230, 70, 30]
    return Observation(
        "rgb-fixture",
        rgb,
        np.ones((128, 128)),
        {
            "image_size": [128, 128],
            "world_bounds_m": [0.15, 0.95, -0.4, 0.4],
            "camera_height_m": 1.305,
        },
    )


def test_spatial_transforms_decode_selection_and_predictor_boundary():
    obs, config = observation(), ReferenceSceneConfig()
    batch = build_action_batch(obs, config)
    assert batch.rgb.shape == (96, 64, 64, 3)
    assert batch.feasible.shape == (96, 64, 64)
    assert batch.feasible.any()
    action = choose_action(batch, rng=np.random.default_rng(7))
    t, v, u = action.encoding["map_index_tvu"]
    np.testing.assert_allclose(batch.encode(action.pixels, t), [u, v], atol=1e-10)
    np.testing.assert_allclose(batch.decode(t, u, v), action.pixels)

    # A fake predictor receives only RGB, and infeasible large values cannot win.
    def predictor(rgb):
        assert rgb.dtype == np.uint8 and rgb.shape[-1] == 3
        values = np.full(batch.feasible.shape, 100.0)
        values[batch.feasible] = -1.0
        values[t, v, u] = 1.0
        return values

    assert choose_action(batch, predictor=predictor).pixels == action.pixels
    with pytest.raises(ValueError):
        choose_action(batch, predictor=lambda _: np.full((96, 64, 64), np.nan))
    batch.feasible[:] = False
    assert choose_action(batch, rng=np.random.default_rng(7)) is None


def test_reference_budget_durable_limits(tmp_path):
    budget = ReferenceBudget(tmp_path)
    for _ in range(6):
        with budget.reserve("diagnostic", 1):
            pass
    with pytest.raises(RuntimeError, match="attempt budget"):
        with budget.reserve("diagnostic", 1):
            pass
    with pytest.raises(RuntimeError):
        with budget.reserve("preparation", 4000):
            pass
    budget.claim_episode()
    budget.claim_episode()
    with pytest.raises(RuntimeError):
        budget.claim_episode()
    with pytest.raises(ValueError):
        with budget.reserve("contact", 1):
            raise ValueError("uncertain")
    document = json.loads((tmp_path / "reference-budget.json").read_text())
    assert document["operations"][-1]["charged_seconds"] == 1
    assert document["operations"][-1]["status"] == "failed_or_uncertain"


class FakeReference:
    """Exact prescribed translation, deliberately no simulated physical evidence."""

    def __init__(self):
        self.config = ReferenceSceneConfig()
        self.rest_positions, self.faces = square_mesh(0.4, 0.1)
        self.edges = mesh_edges(self.faces)
        self.rest_edge_lengths = np.linalg.norm(
            np.diff(self.rest_positions[self.edges], axis=1)[:, 0], axis=1
        )
        self.p = self.rest_positions + [0.5, 0, 0.311]
        self.p[:, 0] = 0.5 + self.rest_positions[:, 0] * 0.5
        self.v = np.zeros_like(self.p)
        self.attached_ids = ()
        self.simulated_time = 0.0
        self.particle_diameter = 0.01
        self.tracking_error_m = 0.0
        self.release_evidence = {}
        self.history = []

    def observe(self):
        return Observation(str(self.simulated_time), np.zeros((16, 16, 3), dtype=np.uint8))

    def snapshot(self):
        return {"positions": self.p.copy(), "free": self._free()}

    def coverage(self):
        return 0.8, 0.192, 0.24

    def attach(self, pixels):
        self.attached_ids = (0, 20)
        return self.attached_ids

    def get_positions(self):
        return self.p.copy()

    def get_velocities(self):
        return self.v.copy()

    def _free(self):
        free = np.ones(len(self.p), dtype=bool)
        free[list(self.attached_ids)] = False
        return free

    def set_attachment_targets(self, targets):
        old = self.p.copy()
        self.p += targets.mean(axis=0) - self.p[list(self.attached_ids)].mean(axis=0)
        self.p[list(self.attached_ids)] = targets
        self.v = (self.p - old) / self.config.sim_dt

    def step(self):
        self.history.append(tuple(self.attached_ids))
        self.simulated_time += self.config.sim_dt
        if not self.attached_ids:
            self.v[:] = 0

    def release(self):
        if self.attached_ids:
            self.release_evidence = {"passed": True, "velocity_error_m_s": 0.0}
        self.attached_ids = ()

    def take_contact_accounting(self):
        return {"available": False}


def test_motion_retains_pins_through_drag_and_certifies_cleanup():
    from cloth_fling.contracts import GraspAction

    env = FakeReference()
    cleanup = []
    outcome = execute_reference_action(
        env, GraspAction(((0, 0), (1, 1))), cleanup_callback=cleanup.append
    )
    assert outcome.execution_valid, outcome.reason
    assert outcome.diagnostics["settled"]
    assert cleanup[0]["passed"]
    commands = outcome.evidence["commands"]
    assert {c["phase"] for c in commands} >= {
        "lift",
        "tauten",
        "airborne_hold",
        "laydown",
        "drag",
        "release",
        "settle",
    }
    assert all(len(c["attachment_ids"]) == 2 for c in commands if c["phase"] == "drag")
    assert all(not c["attachment_ids"] for c in commands if c["phase"] in ("release", "settle"))
    assert outcome.diagnostics["airborne_hold_seconds"] == pytest.approx(3)
    assert outcome.diagnostics["release_tcp_speed_m_s"] < 0.01
    assert outcome.simulated_seconds <= 35
    assert not placement_passed(outcome)  # Fake geometry containment is not a qualification.


def test_reference_interfaces_import_offline():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import cloth_fling.spatial_actions, cloth_fling.reference_run; "
            "assert 'genesis' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_learning_view_excludes_privileged_fields_and_invalid_rewards(tmp_path):
    from cloth_fling.contracts import ActionOutcome, GraspAction
    from cloth_fling.data import DatasetWriter
    from cloth_fling.spatial_actions import learning_samples

    before = observation()
    after = replace(before, observation_id="after")
    for number, (valid, settled) in enumerate(((True, True), (False, True), (True, False))):
        root = tmp_path / str(number)
        outcome = ActionOutcome(
            before,
            after,
            GraspAction(((30, 40), (70, 40))),
            0.8,
            0.6,
            0.192,
            0.144,
            0.24,
            valid,
            "test",
            1,
            0.1,
            truncated=True,
            diagnostics={"settled": settled},
            evidence={
                "selection": {
                    "encoding": {"codec": "test"},
                    "valid_action_mask": np.ones((96, 64, 64), dtype=bool),
                },
                "native_positions": np.zeros((10, 3)),
            },
        )
        with DatasetWriter(root) as writer:
            eid = writer.save_reset(
                before,
                seed=7,
                config=ReferenceSceneConfig(),
                state={"positions": np.zeros((10, 3)), "attached_ids": []},
            )
            writer.begin_attempt(eid, outcome.action, before=before)
            writer.append_outcome(eid, outcome)
        samples = list(learning_samples(root))
        assert len(samples) == int(valid and settled)
        if samples:
            assert set(samples[0]) == {
                "rgb",
                "valid_action_mask",
                "action",
                "reward",
                "split",
                "reset_group_id",
            }
            assert samples[0]["reward"] == pytest.approx(-0.2)


def test_budget_predecessor_preserves_spent_attempts(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    budget = ReferenceBudget(previous)
    for _ in range(6):
        with budget.reserve("diagnostic", 1):
            pass
    (previous / "reference-report.json").write_text('{"status":"placement_limitation"}')
    next_root = tmp_path / "next"
    next_root.mkdir()
    carried = ReferenceBudget(next_root, previous=previous)
    with pytest.raises(RuntimeError, match="attempt budget"):
        with carried.reserve("diagnostic", 1):
            pass
    document = json.loads((next_root / "reference-budget.json").read_text())
    assert document["predecessor"]["budget_sha256"]
    assert len(document["operations"]) == 6


def test_native_release_preserves_velocity_even_when_api_clears_it():
    from cloth_fling.sim.reference import ReferenceBackend

    env = object.__new__(ReferenceBackend)
    env._closed = False
    env.attached_ids = (0, 2)
    env._targets = np.ones((2, 3))
    env._observation = None
    velocities = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]], dtype=float)
    expected = velocities.copy()
    free = np.array([False, True, False])

    class Native:
        def release_particle(self, indices):
            velocities[indices] = 0
            free[indices] = True

        def set_particles_vel(self, values, indices):
            velocities[indices] = values

    env.cloth = Native()
    env.get_velocities = lambda: velocities.copy()
    env._free = lambda: free.copy()
    env.release()
    np.testing.assert_array_equal(velocities, expected)
    assert not env.attached_ids
    assert env.release_evidence["passed"]
    assert env.release_evidence["velocity_error_m_s"] == 0


def test_native_ownership_or_cleanup_failure_invalidates_action():
    from cloth_fling.contracts import GraspAction

    env = FakeReference()
    env._free = lambda: np.zeros(len(env.p), dtype=bool)
    certificates = []
    outcome = execute_reference_action(
        env, GraspAction(((0, 0), (1, 1))), cleanup_callback=certificates.append
    )
    assert not outcome.execution_valid
    assert outcome.reason == "cleanup_failed"
    assert "native_ownership_loss" in outcome.evidence["failure"]
    assert certificates and certificates[0]["passed"] is False
