"""Offline position-friction, progressive-deposition and campaign guard checks."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_reference_tune import NativeFake

from cloth_fling.contracts import GraspAction, ReferenceSceneConfig, scene_config_from_dict
from cloth_fling.reference_diagonal_motion import (
    DepositedBody,
    DiagonalMotion,
    DiagonalTrajectory,
    acceptance,
    diagonal_grid,
    diagonal_schedule,
    execute_diagonal_action,
)
from cloth_fling.reference_diagonal_run import (
    DiagonalBudget,
    _exclusive_claim,
    run_contact_qualification,
    run_diagonal_campaign,
)
from cloth_fling.reference_surface import (
    contact_cases,
    diagonal_scene,
)
from cloth_fling.sim.position_contact import position_coulomb


class DemonstrationFake(NativeFake):
    def __init__(self):
        super().__init__()
        coupler = SimpleNamespace(
            take_particle_contact=lambda **kwargs: {"values": np.zeros((len(self.p), 9))}
        )
        self.scene = SimpleNamespace(sim=SimpleNamespace(coupler=coupler))


@pytest.mark.parametrize("mode", ["qualification", "demonstration"])
def test_deposited_slip_is_observational_only_in_demonstration(mode):
    env = DemonstrationFake()
    run = DiagonalMotion(env, execution_mode=mode)
    run.ids = env.attach(None)
    run.target = env.p[list(run.ids)].copy()
    free = env._free()
    run.body.landed[free] = True
    run.body.cohorts = [np.flatnonzero(free)]
    run.body.anchors = env.p[:, :2] - [0.02, 0]
    if mode == "qualification":
        with pytest.raises(RuntimeError, match="deposited_body_sliding"):
            run.sample("diagonal")
    else:
        run.sample("diagonal")
        assert "deposited_body_sliding" in run.placement_violations
        assert run.body.main_max_p95 == pytest.approx(0.02)


def test_demonstration_finishes_release_and_settling_without_placement_acceptance():
    class Unsettled(DemonstrationFake):
        def step(self):
            super().step()
            if not self.attached_ids:
                self.v[:, 0] = 0.02

    env = Unsettled()
    cleanup = []
    value = execute_diagonal_action(
        env,
        GraspAction(((0, 0), (1, 1))),
        DiagonalTrajectory(),
        expected_ids=(0, 20),
        cleanup_callback=cleanup.append,
        execution_mode="demonstration",
    )
    assert value.execution_valid and value.reason == "completed"
    assert not value.diagnostics["settled"]
    assert value.diagnostics["execution_mode"] == "demonstration"
    assert {
        "insufficient_progressive_deposition",
        "insufficient_deposited_body",
        "settling_timeout",
    } <= value.evidence["placement_violations"].keys()
    phases = {p["phase"] for p in value.evidence["phases"]}
    assert {"diagonal", "endpoint_hold", "release", "settle"} <= phases
    assert cleanup[0]["passed"] and not env.attached_ids
    assert value.simulated_seconds <= 35
    assert not all(acceptance(value).values())


@pytest.mark.parametrize("problem", ["nonfinite", "domain", "strain", "ownership", "deadline"])
def test_demonstration_keeps_physical_execution_stops(problem):
    env = DemonstrationFake()
    run = DiagonalMotion(env, execution_mode="demonstration")
    run.ids = env.attach(None)
    run.target = env.p[list(run.ids)].copy()
    expected = {
        "nonfinite": "nonfinite_state",
        "domain": "solver_domain_escape",
        "strain": "gross_edge_ratio",
        "ownership": "native_ownership_loss",
        "deadline": "reference_simulated_deadline",
    }[problem]
    if problem == "nonfinite":
        env.v[1, 0] = np.nan
    elif problem == "domain":
        env.p += [2, 0, 0]
    elif problem == "strain":
        env.p[0, 0] = 1.1
    elif problem == "ownership":
        env.attached_ids = ()
    else:
        env.simulated_time = 36
    with pytest.raises(RuntimeError, match=expected):
        run.sample("diagonal")


def test_demonstration_observes_edge_overhang_and_rejects_unknown_mode():
    env = DemonstrationFake()
    run = DiagonalMotion(env, execution_mode="demonstration")
    env.p += [0.4, 0, 0]
    run.sample("endpoint_hold")
    assert "placement_far_edge_crossing" in run.placement_violations
    with pytest.raises(ValueError, match="execution mode"):
        DiagonalMotion(env, execution_mode="anything")


def test_position_friction_changes_same_step_position_and_consistent_velocity():
    initial = np.array([[0, 0, 0.305]] * 4, dtype=float)
    predicted = initial + np.array(
        [[0.001, 0, -0.01], [0.01, 0, -0.01], [0.01, 0, -0.01], [0.01, 0, 0]]
    )
    native = predicted.copy()
    native[:, 2] = 0.305
    final, v, correction = position_coulomb(
        initial, predicted, native, free=[True, True, False, True], dt=0.01
    )
    assert final[0, 0] == 0
    assert final[1, 0] == pytest.approx(0.0055)
    np.testing.assert_array_equal(final[2:], native[2:])
    np.testing.assert_allclose(v, (final - initial) / 0.01)
    np.testing.assert_array_equal(final[:, 2], native[:, 2])
    assert np.linalg.norm(correction[1]) <= 0.45 * 0.01 + 1e-12
    zero = position_coulomb(
        initial, predicted, native, free=[True] * 4, dt=0.01, static=0, kinetic=0
    )[0]
    np.testing.assert_array_equal(zero, native)


def test_random_position_contact_coulomb_dissipation_and_input_validation():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(1000, 3))
    a[:, 2] = 0.305
    predicted = a + rng.normal(scale=0.01, size=a.shape)
    predicted[:, 2] = 0.305 - rng.uniform(0, 0.01, len(a))
    native = predicted.copy()
    native[:, 2] = 0.305
    final, v, c = position_coulomb(a, predicted, native, free=np.ones(len(a), bool), dt=0.001)
    assert (np.linalg.norm(c, axis=1) <= 0.45 * (native[:, 2] - predicted[:, 2]) + 1e-12).all()
    before = (native - a) / 0.001
    assert (np.sum(v * v, axis=1) <= np.sum(before * before, axis=1) + 1e-9).all()
    assert not c[:, 2].any()
    for kwargs in ({"dt": 0}, {"dt": np.nan}, {"dt": 0.001, "static": 0.3, "kinetic": 0.4}):
        with pytest.raises(ValueError):
            position_coulomb(a, predicted, native, free=np.ones(len(a), bool), **kwargs)


@pytest.mark.parametrize("dt", [0.001, 0.0005])
@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_known_load_sticks_or_breaks_away_at_coulomb_threshold(dt, factor):
    p = np.array([[0.5, 0, 0.305]], float)
    v = np.zeros_like(p)
    start = p.copy()
    for _ in range(round(0.2 / dt)):
        initial = p.copy()
        v += np.array([factor * 0.45 * 9.81, 0, -9.81]) * dt
        predicted = p + v * dt
        native = predicted.copy()
        native[:, 2] = 0.305
        p, v, _ = position_coulomb(initial, predicted, native, free=[True], dt=dt)
    if factor < 1:
        np.testing.assert_allclose(p, start, atol=1e-12)
        np.testing.assert_allclose(v, 0, atol=1e-10)
    else:
        assert p[0, 0] > start[0, 0] + 0.04
        assert v[0, 0] == pytest.approx((factor - 1) * 0.45 * 9.81 * 0.2, rel=1e-8)


def test_versioned_profiles_do_not_inherit_parent_identity_or_defaults():
    old = ReferenceSceneConfig()
    new = diagonal_scene()
    assert scene_config_from_dict(new.to_dict()) == new
    a, b = old.to_dict(), new.to_dict()
    assert {k for k in a if a[k] != b[k]} == {"profile", "contact_profile"}
    assert old.contact_profile == "reference_split_v1"
    assert len(contact_cases()) == 16
    assert len({x["name"] for x in contact_cases()}) == 16


def test_diagonal_grid_stays_straight_and_reserves_worst_case_tug():
    grid = diagonal_grid()
    assert [(c.endpoint_x_m, c.diagonal_seconds) for c in grid] == [
        (x, t) for x in (0.78, 0.76, 0.74) for t in (6, 8, 10, 4)
    ]
    for c in grid:
        segments, schedule = diagonal_schedule(c)
        s = segments[0]
        times = np.linspace(0, s.seconds, 200)
        p = s.at(times)
        np.testing.assert_allclose(
            (p[0] - 0.1) / (c.endpoint_x_m - 0.1), (0.95 - p[2]) / 0.64, atol=1e-12
        )
        assert s.minimum_horizontal_speed() >= -1e-10
        assert s.maximum_norm(1) <= 0.8 and s.maximum_norm(2) <= 2
        assert schedule["maximum_action_seconds"] <= 35 + 1e-9
        assert 0 < schedule["tauten_allowance_seconds"] <= 20
        for d in (1, 2):
            np.testing.assert_allclose(s.at(0, d), 0, atol=1e-12)
            np.testing.assert_allclose(s.at(s.seconds, d), 0, atol=1e-12)


def test_deposited_ids_and_anchors_cannot_hide_motion_by_losing_contact():
    body = DepositedBody(20)
    p = np.tile([0.3, 0, 0.305], (20, 1))
    v = np.zeros_like(p)
    free = np.ones(20, bool)
    for _ in range(5):
        body.update(p, v, free, np.ones(20, bool), [0.7, 0, 0.5], "diagonal")
    body.update(p, v, free, np.ones(20, bool), [0.7, 0, 0.39], "diagonal")
    assert body.at_height == 1
    body.freeze_tug(p, free)
    p[:, 0] += 0.02
    body.update(p, v, free, np.zeros(20, bool), [0.7, 0, 0.31], "tug")
    assert len(body.tug_ids) == 20
    assert body.tug_max_median == pytest.approx(0.02)
    body.update(p, v, free, np.zeros(20, bool), [0.7, 0, 0.31], "diagonal")
    assert body.landed.all() and body.main_max_p95 == pytest.approx(0.02)


@pytest.mark.parametrize(
    "mode,expected_n,reason",
    [
        ("cap", 20, "distance_cap"),
        ("taut", 0, "geometric_slack_removed"),
        ("slip", 2, "incipient_body_slip"),
    ],
)
def test_adaptive_tug_stops_at_rest_without_reclassifying_body(mode, expected_n, reason):
    env = NativeFake()
    run = DiagonalMotion(env)
    run.ids = env.attach(None)
    run.target = np.array([[0.78, -0.2, 0.31], [0.78, 0.2, 0.31]])
    env.p[:] = [0.4, 0, 0.305]
    env.p[list(run.ids)] = run.target
    run.hold = lambda *args: None
    run.body.stable[:] = 5
    run.slack_stable = 5 if mode == "taut" else 0
    calls = []

    def curve(segment):
        calls.append(segment)
        run.target += np.array([0.001, 0, 0])
        if mode == "slip":
            env.p[env._free(), 0] += 0.0006
        for d in (1, 2):
            np.testing.assert_allclose(segment.at(segment.seconds, d), 0, atol=1e-9)

    run.curve = curve
    run.perform_tug()
    assert len(calls) == expected_n
    assert run.tug_stop_reason == reason
    assert run.tug_distance <= 0.02 + 1e-12


def test_budget_stage_continuation_and_uncertain_reservations(tmp_path):
    budget = DiagonalBudget(tmp_path, create=True)
    for _ in range(16):
        with budget.reserve("contact", 1):
            assert (
                json.loads((tmp_path / "diagonal-budget.json").read_text())["operations"][-1][
                    "status"
                ]
                == "reserved"
            )
    with pytest.raises(RuntimeError, match="attempt budget"):
        with budget.reserve("contact", 1):
            pass
    continued = DiagonalBudget(tmp_path)
    assert len(continued.document["operations"]) == 16
    for _ in range(12):
        with continued.reserve("development", 1):
            pass
    with pytest.raises(RuntimeError, match="attempt budget"):
        with continued.reserve("development", 1):
            pass
    for _ in range(3):
        with continued.reserve("confirmation", 1):
            pass
    with pytest.raises(RuntimeError, match="attempt budget"):
        with continued.reserve("confirmation", 1):
            pass
    with pytest.raises(ValueError):
        with continued.reserve("preparation", 5):
            raise ValueError("uncertain")
    with pytest.raises(RuntimeError, match="cannot resume"):
        DiagonalBudget(tmp_path)


def test_contact_gate_and_terminal_stages_never_open_a_scene(tmp_path):
    assert run_diagonal_campaign(tmp_path)["status"] == "contact_gate_stop"
    (tmp_path / "contact-report.json").write_text(
        '{"status":"contact_limitation","stage_passed":false}'
    )
    assert run_diagonal_campaign(tmp_path)["status"] == "contact_gate_stop"
    assert (
        run_contact_qualification(tmp_path, fixture_dir="missing")["status"] == "contact_limitation"
    )
    (tmp_path / "diagonal-report.json").write_text('{"status":"started"}')
    assert run_diagonal_campaign(tmp_path)["status"] == "uncertain_incomplete"
    assert not (tmp_path / "gpu-lease.json").exists()


def test_claim_is_exclusive_and_durable(tmp_path):
    path = tmp_path / "claim"
    _exclusive_claim(path, {"state": "fixed"})
    assert json.loads(path.read_text()) == {"state": "fixed"}
    with pytest.raises(FileExistsError):
        _exclusive_claim(path, {"state": "retry"})


def test_selection_failure_still_certifies_cleanup():
    cert = []
    value = execute_diagonal_action(
        NativeFake(),
        GraspAction(((0, 0), (1, 1))),
        DiagonalTrajectory(),
        expected_ids=(3, 4),
        cleanup_callback=cert.append,
    )
    assert value.reason == "favorable_vertex_identity_mismatch"
    assert cert[0]["passed"]


def test_new_modules_are_offline():
    script = "import sys; import cloth_fling.reference_diagonal_run, cloth_fling.sim.position_contact; assert 'genesis' not in sys.modules and 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_parent_fixture_import_is_explicit_and_rejects_changed_material():
    from dataclasses import replace

    from cloth_fling.reference_surface import import_parent_fixture
    from cloth_fling.sim.backend import _state_hash

    class Env:
        def __init__(self):
            self.config = diagonal_scene()
            self.faces = np.array([[0, 1, 2]])
            self.edges = np.array([[0, 1], [1, 2], [0, 2]])
            self.rest_positions = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float32)
            self.p = self.rest_positions + [0.5, 0, 0.311]
            self.v = np.zeros_like(self.p)
            self.cloth = self
            self.reset_metadata = {}

        def release(self):
            pass

        def release_particle(self):
            pass

        def set_particles_pos(self, p):
            self.p = p.copy()

        def set_particles_vel(self, v):
            self.v = v.copy()

        def _dirty(self):
            pass

        def snapshot(self):
            value = {
                "positions": self.p,
                "velocities": self.v,
                "free": np.ones(3, bool),
                "config_json": json.dumps(self.config.to_dict()),
                "schema_version": 2,
                "faces": self.faces,
                "edges": self.edges,
                "rest_positions": self.rest_positions,
                "simulated_time": 0.0,
                "reset_metadata_json": "{}",
            }
            value["state_hash"] = _state_hash(value)
            return value

    env = Env()
    parent = env.snapshot()
    parent["config_json"] = json.dumps(ReferenceSceneConfig().to_dict())
    parent["state_hash"] = _state_hash(parent)
    child = import_parent_fixture(env, parent)
    assert child["state_hash"] != parent["state_hash"]
    np.testing.assert_array_equal(child["positions"], parent["positions"])
    env.config = replace(diagonal_scene(), static_friction=0.7)
    with pytest.raises(ValueError, match="contact identity only"):
        import_parent_fixture(env, parent)
    env.config = diagonal_scene()
    parent["positions"] = parent["positions"] + [0.01, 0, 0]
    with pytest.raises(ValueError, match="invalid parent"):
        import_parent_fixture(env, parent)


def test_contact_failure_stops_before_any_motion_and_cannot_extend(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from test_reference_tune import archived_campaign

    import cloth_fling.reference_diagonal_run as run
    from cloth_fling.sim.reference import ReferenceBackend

    source = tmp_path / "history"
    archived_campaign(source)
    root = tmp_path / "campaign"

    class Env:
        def __init__(self, *a, **k):
            self.attached_ids = ()
            self.obs = type(
                "Obs",
                (),
                {
                    "rgb": np.zeros((2, 2, 3), np.uint8),
                    "depth": np.zeros((2, 2)),
                    "observation_id": "exact",
                },
            )()

        def observe(self):
            return self.obs

        def release(self):
            pass

        def _free(self):
            return np.ones(3, bool)

        def close(self):
            pass

    monkeypatch.setattr(ReferenceBackend, "__new__", staticmethod(lambda cls, *a, **k: Env()))
    monkeypatch.setattr(run, "reference_lease", lambda root: nullcontext())
    monkeypatch.setattr(run, "import_parent_fixture", lambda *a: {"state_hash": "new"})
    monkeypatch.setattr(run, "restore_exact", lambda *a: None)
    original = run.save_bundle
    monkeypatch.setattr(
        run,
        "save_bundle",
        lambda root, name, value: (
            original(root, name, {"test": "observation"})
            if name == "fixture-observation"
            else original(root, name, value)
        ),
    )
    calls = []

    def coupon(config, case, **kwargs):
        calls.append(case["name"])
        kwargs["cleanup_callback"]({"passed": True})
        return {"summary": {"passed": False, "gates": {"sticking": False}}}

    monkeypatch.setattr(run, "run_contact_case", coupon)
    report = run_contact_qualification(root, fixture_dir=source)
    assert report["status"] == "contact_limitation", report
    assert calls == ["rest"]
    assert run_diagonal_campaign(root)["status"] == "contact_gate_stop"
    assert run_contact_qualification(root, fixture_dir="missing")["status"] == "contact_limitation"
    with pytest.raises(FileExistsError):
        run_contact_qualification(tmp_path / "extension", fixture_dir=source)
    assert calls == ["rest"]


def test_contact_only_failure_replay_preserves_native_geometry_and_hashes(tmp_path):
    from test_replay import snapshot

    from cloth_fling.reference_run import save_bundle
    from cloth_fling.replay import load_replay

    state = snapshot()
    p = state["positions"]
    ref = save_bundle(
        tmp_path,
        "contact-00",
        {
            "initial_state": state,
            "positions": np.stack([p, p + [0.002, 0, 0]]),
            "velocities": np.zeros((2, *p.shape)),
            "times": np.array([0.0, 0.5]),
            "summary": {"passed": False},
        },
    )
    (tmp_path / "contact-report.json").write_text(
        json.dumps({"cases": [{"name": "rest", "outcome": ref}]})
    )
    actions, origin = load_replay(tmp_path)
    assert origin["format"] == "reference_diagonal" and not origin["motion_evaluated"]
    assert len(actions) == 1 and actions[0].execution_valid is False
    assert actions[0].traces[-1]["simulated_seconds"] == 0.5
    np.testing.assert_array_equal(actions[0].traces[-1]["cloth_positions_m"], p + [0.002, 0, 0])
    (tmp_path / ref["path"]).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_replay(tmp_path)


def test_completed_demonstration_replay_does_not_label_quality_metrics_as_failed_execution(
    tmp_path,
):
    from test_replay import snapshot

    from cloth_fling.reference_run import save_bundle
    from cloth_fling.replay import load_replay

    state = snapshot()
    fixture = save_bundle(tmp_path, "fixture", state)
    outcome = save_bundle(
        tmp_path,
        "action-00",
        {
            "simulated_seconds": 0.1,
            "execution_valid": True,
            "reason": "completed",
            "diagnostics": {"execution_mode": "demonstration"},
            "evidence": {
                "phase_traces": [
                    {
                        "phase": "settle",
                        "simulated_seconds": 0,
                        "cloth_positions_m": state["positions"],
                        "attachment_ids": [],
                    }
                ]
            },
        },
    )
    (tmp_path / "contact-report.json").write_text(json.dumps({"cases": [], "fixture": fixture}))
    (tmp_path / "diagonal-report.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "attempt": 0,
                        "category": "demonstration",
                        "candidate": 0,
                        "passed": False,
                        "outcome": outcome,
                    }
                ]
            }
        )
    )
    actions, _ = load_replay(tmp_path)
    assert actions[0].execution_valid
    assert "completed" in actions[0].label and "unqualified" in actions[0].label
    assert "FAIL" not in actions[0].label


def test_campaign_configuration_matches_compiler_and_case_order():
    from dataclasses import asdict
    from pathlib import Path

    root = Path(__file__).parents[1]
    doc = json.loads((root / "configs/reference-diagonal-campaign-v3.json").read_text())
    assert doc["contact_cases"] == contact_cases()
    assert doc["candidates"] == [
        {"index": i, "trajectory": asdict(c), "schedule": diagonal_schedule(c)[1]}
        for i, c in enumerate(diagonal_grid())
    ]


def test_new_landings_do_not_dilute_a_previously_deposited_cohort():
    body = DepositedBody(100)
    p = np.tile([0.3, 0, 0.305], (100, 1))
    v = np.zeros_like(p)
    free = np.ones(100, bool)
    contacts = np.zeros(100, bool)
    contacts[0] = True
    for _ in range(5):
        body.update(p, v, free, contacts, [0.7, 0, 0.5], "diagonal")
    contacts[:] = True
    for _ in range(4):
        body.update(p, v, free, contacts, [0.7, 0, 0.5], "diagonal")
    p[0, 0] += 0.02
    body.update(p, v, free, contacts, [0.7, 0, 0.5], "diagonal")
    assert body.main_max_p95 == pytest.approx(0.02)
    assert len(body.cohorts) == 2


def test_airborne_coupon_finishes_before_ballistic_table_contact():
    case = next(c for c in contact_cases() if c["kind"] == "airborne")
    dt = 0.001
    duration = case["duration_s"]
    # Semi-implicit Euler includes the extra t*dt term; use the half-dt check too.
    for step in (dt, dt / 2):
        height = 0.8 - 0.5 * 9.81 * duration * (duration + step)
        assert height > 0.30
    assert duration <= 0.5


def test_corrected_airborne_fixture_does_not_reopen_terminal_campaign(tmp_path):
    (tmp_path / "contact-report.json").write_text(
        '{"status":"contact_limitation","stage_passed":false}'
    )
    assert (
        run_contact_qualification(tmp_path, fixture_dir="missing")["status"] == "contact_limitation"
    )
    assert run_diagonal_campaign(tmp_path)["status"] == "contact_gate_stop"
    assert not (tmp_path / "diagonal-stage.claim").exists()
