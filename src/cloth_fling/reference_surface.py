"""Bounded native contact coupons and explicit reference-fixture ancestry."""

from dataclasses import replace

import numpy as np

from cloth_fling.contracts import ReferenceSceneConfig
from cloth_fling.sim.backend import _state_hash


def diagonal_scene(**changes):
    return replace(
        ReferenceSceneConfig(),
        profile="reference_diagonal_v3",
        contact_profile="reference_position_coulomb_v1",
        **changes,
    )


def import_parent_fixture(env, parent):
    """Initialization only: preserve geometry, issue a new physical identity, never forge a restore."""
    if (
        parent["schema_version"] != 2
        or parent["state_hash"] != _state_hash(parent)
        or not np.asarray(parent["free"]).all()
    ):
        raise ValueError("invalid parent native fixture")
    import json

    expected = ReferenceSceneConfig().to_dict()
    if json.loads(parent["config_json"]) != json.loads(json.dumps(expected)):
        raise ValueError("parent physical configuration differs")
    child = env.config.to_dict()
    for key in ("profile", "contact_profile"):
        child[key] = expected[key]
    if child != expected:
        raise ValueError("new fixture may change contact identity only")
    for key in ("faces", "edges", "rest_positions"):
        if not np.array_equal(getattr(env, key), parent[key]):
            raise ValueError("parent native topology/geometry differs")
    env.release()
    env.cloth.release_particle()
    env.cloth.set_particles_pos(parent["positions"])
    env.cloth.set_particles_vel(parent["velocities"])
    env.simulated_time = float(parent["simulated_time"])
    env.reset_metadata = {
        **json.loads(parent["reset_metadata_json"]),
        "parent_state_hash": parent["state_hash"],
        "contact_qualification_inherited": False,
    }
    env._dirty()
    state = env.snapshot()
    for key in ("positions", "velocities", "free"):
        if not np.array_equal(state[key], parent[key]):
            raise RuntimeError("fixture import readback regression")
    if state["state_hash"] == parent["state_hash"]:
        raise RuntimeError("new profile inherited old physical identity")
    return state


def contact_cases():
    return [
        {"name": "rest", "kind": "rest"},
        {"name": "coast_positive", "kind": "coast", "direction": 1},
        {"name": "coast_negative", "kind": "coast", "direction": -1},
        {"name": "uniform_sub", "kind": "load", "mode": "uniform", "factor": 0.5, "direction": 1},
        {"name": "uniform_super", "kind": "load", "mode": "uniform", "factor": 1.5, "direction": 1},
        *[
            {
                "name": f"edge_{level}_{sign}",
                "kind": "load",
                "mode": "edge",
                "factor": factor,
                "direction": direction,
            }
            for level, factor in (("sub", 0.5), ("super", 1.5))
            for sign, direction in (("positive", 1), ("negative", -1))
        ],
        {"name": "oblique_landing", "kind": "landing"},
        {"name": "airborne", "kind": "airborne", "duration_s": 0.3},
        {"name": "attached_exclusion", "kind": "attached"},
        {
            "name": "edge_sub_half_dt",
            "kind": "load",
            "mode": "edge",
            "factor": 0.5,
            "direction": 1,
            "dt": 0.0005,
        },
        {
            "name": "edge_super_half_dt",
            "kind": "load",
            "mode": "edge",
            "factor": 1.5,
            "direction": 1,
            "dt": 0.0005,
        },
        {
            "name": "zero_uniform",
            "kind": "load",
            "mode": "uniform",
            "factor": 0.5,
            "direction": 1,
            "zero": True,
        },
        {
            "name": "zero_edge",
            "kind": "load",
            "mode": "edge",
            "factor": 0.5,
            "direction": 1,
            "zero": True,
        },
    ]


def summarize_case(case, positions, velocities, times, mask, accounting, contact_samples):
    p, v, t = np.asarray(positions), np.asarray(velocities), np.asarray(times)
    full_center = p.mean(axis=1)
    during = (t >= 0.1 - 1e-8) & (t <= 0.3 + 1e-8)
    unloaded = ~mask if case.get("mode") == "edge" else np.ones(len(mask), bool)
    dx = p[during][:, unloaded, :2] - p[np.argmin(abs(t - 0.1)), unloaded, :2]
    drift = np.linalg.norm(dx, axis=2)
    max_p95 = float(np.quantile(drift, 0.95, axis=1).max())
    signed_displacement = float(
        case.get("direction", 1)
        * (full_center[np.argmin(abs(t - 0.3)), 0] - full_center[np.argmin(abs(t - 0.1)), 0])
    )
    totals = accounting["totals"]
    result = {
        "unloaded_drift_max_p95_m": max_p95,
        "signed_load_displacement_m": signed_displacement,
        "terminal_rms_m_s": float(np.sqrt(np.mean(np.sum(v[-1] ** 2, axis=1)))),
        "accounting": accounting,
    }
    gates = {
        "finite": bool(np.isfinite(p).all() and np.isfinite(v).all()),
        "dissipative": totals["positive_friction_work_j"] <= 1e-9,
        "attached_excluded": totals["attached_friction_events"] == 0,
        "coulomb_bound": totals["tangent_impulse_kg_m_s"]
        <= 0.45 * totals["normal_impulse_kg_m_s"] + 1e-7,
    }
    kind = case["kind"]
    if kind == "rest":
        gates["rest_stable"] = (
            float(
                np.quantile(
                    np.linalg.norm(p[-1, :, :2] - p[np.argmin(abs(t - 0.1)), :, :2], axis=1), 0.95
                )
            )
            <= 0.001
        )
    elif kind == "coast":
        speed = case["direction"] * v[:, :, 0].mean(axis=1)
        fit = (t > 0.005) & (speed > 0.05) & (t < 0.1)
        deceleration = -float(np.polyfit(t[fit], speed[fit], 1)[0]) if fit.sum() >= 4 else None
        result["deceleration_m_s2"] = deceleration
        gates["coasting_law"] = (
            deceleration is not None and abs(deceleration - 0.45 * 9.81) / (0.45 * 9.81) <= 0.2
        )
        gates["coasting_stopped"] = abs(float(speed[-1])) <= 0.005
    elif kind == "load":
        sub = case["factor"] < 1 and not case.get("zero")
        gates["loaded_stick" if sub else "loaded_slip"] = (
            max_p95 <= 0.001 if sub else signed_displacement >= 0.005
        )
        if case.get("mode") == "uniform" and case["factor"] > 1:
            fit = (t >= 0.12) & (t <= 0.28)
            acceleration = float(
                np.polyfit(t[fit], case["direction"] * v[fit, :, 0].mean(axis=1), 1)[0]
            )
            result["acceleration_m_s2"] = acceleration
            gates["loaded_acceleration_law"] = (
                abs(acceleration - 0.5 * 0.45 * 9.81) / (0.5 * 0.45 * 9.81) <= 0.2
            )
        if case.get("zero"):
            gates["zero_control_no_friction"] = totals["tangent_impulse_kg_m_s"] == 0
    elif kind == "landing":
        gates["landed_and_slowed"] = (
            totals["free_contacts"] > 0 and result["terminal_rms_m_s"] <= 0.01
        )
    elif kind == "airborne":
        gates["airborne_excluded"] = (
            totals["free_contacts"] == 0 and totals["tangent_impulse_kg_m_s"] == 0
        )
    elif kind == "attached":
        gates["no_attached_correction"] = all(
            not sample["values"][sample["attached_ids"], 2:5].any() for sample in contact_samples
        )
    result.update(gates=gates, passed=all(gates.values()))
    return result


def run_contact_case(config, case, *, backend, cleanup_callback, evidence_callback=None):
    from cloth_fling.sim.appearance import AppearanceConfig
    from cloth_fling.sim.reference import ReferenceBackend

    cfg = replace(
        config, sim_dt=case.get("dt", config.sim_dt), table_friction=not case.get("zero", False)
    )
    env = None
    try:
        env = ReferenceBackend(
            cfg,
            backend=backend,
            appearance=AppearanceConfig(),
            diagnostic_load=case["kind"] == "load",
        )
        p = env.rest_positions.copy()
        p[:, :2] += cfg.cloth_center_m[:2]
        p[:, 2] = 0.305
        if case["kind"] == "landing":
            p[:, 2] += 0.10
        if case["kind"] == "airborne":
            p[:, 2] += 0.8
        v = np.zeros_like(p)
        if case["kind"] == "coast":
            v[:, 0] = 0.3 * case["direction"]
        if case["kind"] in ("landing", "airborne"):
            v[:, 0] = 0.1
        env.cloth.set_particles_pos(p)
        env.cloth.set_particles_vel(v)
        env._dirty()
        ids = []
        if case["kind"] == "attached":
            ids = [
                int(np.argmin(env.rest_positions[:, 0])),
                int(np.argmax(env.rest_positions[:, 0])),
            ]
            env.attached_ids = tuple(ids)
            env._targets = p[ids].copy()
            env.cloth.fix_particles(ids, zero_velocity=False)
        mask = np.ones(len(p), bool)
        if case.get("mode") == "edge":
            x = env.rest_positions[:, 0]
            mask = (
                x <= (x.min() + 0.1 * np.ptp(x))
                if case["direction"] > 0
                else x >= (x.max() - 0.1 * np.ptp(x))
            )
        acceleration = (
            case.get("direction", 1) * case.get("factor", 0) * 0.45 * 9.81 * len(mask) / mask.sum()
        )
        if env.diagnostic_field is not None:
            env.diagnostic_field.mask.from_numpy(mask)
            env.diagnostic_field.acceleration[None] = acceleration
        initial = env.snapshot()
        env.take_contact_accounting(reset=True)
        positions, velocities, times = [env.get_positions()], [env.get_velocities()], [0.0]
        contact_samples = []
        for step in range(round(case.get("duration_s", 0.5) / cfg.sim_dt)):
            sec = step * cfg.sim_dt
            if env.diagnostic_field is not None:
                if 0.1 - 1e-10 <= sec < 0.3 - 1e-10:
                    env.diagnostic_field.activate()
                else:
                    env.diagnostic_field.deactivate()
            env.step()
            p, v = env.get_positions(), env.get_velocities()
            if not np.isfinite(p).all() or not np.isfinite(v).all():
                raise RuntimeError("nonfinite contact coupon state")
            ratio = (
                np.linalg.norm(p[env.edges[:, 0]] - p[env.edges[:, 1]], axis=1)
                / env.rest_edge_lengths
            )
            if ratio.max() >= 3 or (p[:, 2] < -0.1).any() or (abs(p[:, :2]) > 1.2).any():
                raise RuntimeError("contact coupon gross instability")
            positions.append(p)
            velocities.append(v)
            times.append((step + 1) * cfg.sim_dt)
            if (step + 1) % round(0.02 / cfg.sim_dt) == 0:
                contact_samples.append(
                    {
                        "simulated_seconds": times[-1],
                        "attached_ids": ids,
                        **env.scene.sim.coupler.take_particle_contact(),
                    }
                )
        accounting = env.take_contact_accounting()
        summary = summarize_case(
            case, positions, velocities, times, mask, accounting, contact_samples
        )
        return {
            "case": case,
            "config": cfg.to_dict(),
            "initial_state": initial,
            "terminal_state": env.snapshot(),
            "positions": np.asarray(positions),
            "velocities": np.asarray(velocities),
            "times": np.asarray(times),
            "load_mask": mask,
            "external_acceleration_m_s2": acceleration,
            "external_load_active_interval_s": [0.1, 0.3] if case["kind"] == "load" else None,
            "particle_contacts": contact_samples,
            "summary": summary,
        }
    except Exception as exc:
        failure = {
            "case": case,
            "error": f"{type(exc).__name__}: {exc}",
            "positions": np.asarray(positions) if "positions" in locals() else None,
            "velocities": np.asarray(velocities) if "velocities" in locals() else None,
            "times": np.asarray(times) if "times" in locals() else None,
        }
        if env is not None:
            try:
                failure["failure_state"] = env.snapshot()
            except Exception as capture:
                failure["capture_error"] = str(capture)
        if evidence_callback:
            evidence_callback(failure)
        raise
    finally:
        if env is not None:
            try:
                if env.diagnostic_field is not None:
                    env.diagnostic_field.deactivate()
                env.release()
                certificate = {
                    "released": not env.attached_ids,
                    "native_free": bool(env._free().all()),
                }
                certificate["passed"] = all(certificate.values())
                cleanup_callback(certificate)
                if not certificate["passed"]:
                    raise RuntimeError("coupon cleanup regression")
            finally:
                env.close()
