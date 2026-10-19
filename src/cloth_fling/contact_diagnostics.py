"""Bounded, privileged native table-contact diagnostics, independent of primitives.

Run ``python -m cloth_fling.contact_diagnostics --config configs/contact-native-v1.json
--output-dir runs/contact-native-v1``. No rendering, reset shaping or action tuning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from cloth_fling.contracts import SceneConfig


def summarize_contact_trace(positions, velocities, *, dt, radius, mu_kinetic):
    """Specific energies in J/kg; uniform coupon means no guessed particle masses."""
    p, v = np.asarray(positions), np.asarray(velocities)
    t = np.arange(len(p)) * dt
    speed = np.linalg.norm(np.mean(v[:, :, :2], axis=1), axis=1)
    z = p[:, :, 2].mean(axis=1)
    normal_energy = 0.5 * np.mean(v[:, :, 2] ** 2, axis=1) + 9.81 * (z - radius)
    total_energy = 0.5 * np.mean(np.sum(v**2, axis=2), axis=1) + 9.81 * (z - radius)
    supported = np.abs(z - radius) < 2e-5
    sliding = supported & (speed > 0.05) & (t > 0.01)
    slope = -float(np.polyfit(t[sliding], speed[sliding], 1)[0]) if sliding.sum() > 3 else None
    stopped = np.flatnonzero((speed < 0.005) & (t > 0.02))
    end = int(stopped[0]) if len(stopped) else len(t) - 1
    mean_xy = p[:, :, :2].mean(axis=1)
    expected = mu_kinetic * 9.81
    tolerance = max(0.001, 0.02 * abs(float(normal_energy[0])))
    first_contact = int(np.flatnonzero(supported)[0]) if supported.any() else None
    incident = max(first_contact - 1, 0) if first_contact is not None else None
    return {
        "normal_energy_initial_j_kg": float(normal_energy[0]),
        "normal_energy_max_j_kg": float(normal_energy.max()),
        "normal_energy_gain_j_kg": float(normal_energy.max() - normal_energy[0]),
        "normal_energy_tolerance_j_kg": tolerance,
        "normal_energy_passed": bool(normal_energy.max() <= normal_energy[0] + tolerance),
        "total_energy_initial_j_kg": float(total_energy[0]),
        "total_energy_max_j_kg": float(total_energy.max()),
        "total_energy_passed": bool(total_energy.max() <= total_energy[0] + tolerance),
        "first_supported_time_s": float(t[first_contact]) if first_contact is not None else None,
        "incident_mean_velocity_m_s": v[incident].mean(axis=0).tolist()
        if incident is not None
        else None,
        "first_supported_mean_velocity_m_s": v[first_contact].mean(axis=0).tolist()
        if first_contact is not None
        else None,
        "first_supported_total_energy_j_kg": float(total_energy[first_contact])
        if first_contact is not None
        else None,
        "max_upward_speed_m_s": float(v[:, :, 2].max()),
        "max_rebound_height_m": float(max(0, z[np.argmax(supported) :].max() - radius))
        if supported.any()
        else None,
        "deceleration_m_s2": slope,
        "expected_mu_g_m_s2": expected,
        "deceleration_relative_error": abs(slope - expected) / expected
        if slope is not None and expected > 0
        else None,
        "stopped": bool(len(stopped)),
        "stopping_distance_m": float(np.linalg.norm(mean_xy[end] - mean_xy[0])),
        "final_rms_speed_m_s": float(np.sqrt(np.mean(np.sum(v[-1] ** 2, axis=1)))),
        "min_height_m": float(p[:, :, 2].min()),
        "max_height_m": float(p[:, :, 2].max()),
        "max_horizontal_velocity_change_m_s": float(
            np.max(np.linalg.norm(v[:, :, :2] - v[0, :, :2], axis=2))
        ),
    }


def run_contact_diagnostic(
    config, *, output_dir, backend="gpu", duration_s=0.6, include_half_dt=True
):
    """Return JSON report; save every full-step state to NPZ and report.json.

    Accept SceneConfig or its dictionary. Native-only. Serial exclusive scenes.
    Fail the gate and stop further scenarios immediately on normal energy gain.
    This is an unforced flat coupon test, not a reset or primitive qualification.
    """
    from cloth_fling.sim.backend import GenesisBackend

    config = config if isinstance(config, SceneConfig) else SceneConfig(**config)
    if config.contact_profile != "native_coulomb_v1":
        raise ValueError("diagnostics require native_coulomb_v1")
    if not np.isfinite(duration_s) or not 0.3 <= duration_s <= 2:
        raise ValueError("duration_s must be in [.3, 2]")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "config": config.to_dict(),
        "backend": backend,
        "material_claim": "engineering priors; not measured terry/desk coefficients",
        "cases": {},
        "qualification_passed": False,
    }
    cases = [
        ("rest", 0.0, 0.0, False),
        ("slide_positive", 0.5, 0.0, False),
        ("slide_negative", -0.5, 0.0, False),
        ("horizontal_drop", 0.0, 0.10, False),
        ("oblique_landing", 0.5, 0.10, False),
        ("attached", 0.5, 0.0, True),
        ("airborne", 0.5, 0.8, False),
    ]
    try:
        for factor in [1.0, 0.5] if include_half_dt else [1.0]:
            cfg = replace(config, sim_dt=config.sim_dt * factor)
            with GenesisBackend(cfg, backend=backend) as env:
                if factor == 1.0:
                    import genesis as gs
                    import torch

                    from cloth_fling.sim.native_contact import _SOURCE_HASHES

                    report["runtime"] = {
                        "genesis": gs.__version__,
                        "device": str(gs.device),
                        "particle_diameter_m": env.particle_diameter,
                        "substeps": env.substeps,
                        "native_source_sha256_ast": dict(_SOURCE_HASHES),
                        "device_name": torch.cuda.get_device_name(gs.device)
                        if gs.device.type == "cuda"
                        else "cpu",
                    }
                radius = env.particle_diameter / 2
                for name, vx, height, attached in cases:
                    if factor != 1.0 and name != "slide_positive":
                        continue
                    key = name if factor == 1.0 else name + "_half_dt"
                    env.cloth.release_particle()
                    env.attached_ids = ()
                    p = env.rest_positions.copy()
                    p[:, 2] = radius + height
                    v = np.tile([vx, 0.0, 0.0], (len(p), 1))
                    env.cloth.set_particles_pos(p)
                    env.cloth.set_particles_vel(v)
                    env.take_contact_accounting(reset=True)
                    env.scene.sim.coupler.clear_contact_trace()
                    if attached:
                        # Diagnostic fixture only: fix the coupon, retaining native velocity.
                        env.cloth.fix_particles(list(range(len(p))), zero_velocity=False)
                    ps, vs = [env.get_positions()], [env.get_velocities()]
                    seconds = min(duration_s, 0.1) if name == "airborne" else duration_s
                    for _ in range(int(np.ceil(seconds / cfg.sim_dt))):
                        env.step()
                        ps.append(env.get_positions())
                        vs.append(env.get_velocities())
                    ps, vs = np.array(ps), np.array(vs)
                    native = env.scene.sim.coupler.read_contact_trace()
                    cumulative = env.take_contact_accounting()
                    np.savez_compressed(
                        output / (key + ".npz"),
                        positions=ps,
                        velocities=vs,
                        dt=cfg.sim_dt,
                        config_json=json.dumps(cfg.to_dict()),
                        native_contact_columns=np.array(native["columns"]),
                        native_contact_values=native["values"],
                    )
                    metrics = summarize_contact_trace(
                        ps, vs, dt=cfg.sim_dt, radius=radius, mu_kinetic=cfg.table_coefficients[1]
                    )
                    metrics["native_contact"] = cumulative["totals"]
                    metrics["native_cumulative_substeps"] = cumulative["substeps"]
                    metrics["native_trace_substeps"] = native["substeps"]
                    metrics["native_trace_truncated"] = native["truncated"]
                    metrics["native_coulomb_bound_passed"] = bool(
                        np.all(
                            native["values"][:, 1]
                            <= cfg.table_coefficients[0] * native["values"][:, 9] + 1e-7
                        )
                    )
                    report["cases"][key] = metrics
                    # Attached horizontal travel is prescribed; no normal energy source.
                    if not metrics["normal_energy_passed"]:
                        report["stopped_reason"] = (
                            "unresolved native normal energy injection: " + key
                        )
                        return report
        c = report["cases"]
        gates = {
            "native_accounting_complete": all(not m["native_trace_truncated"] for m in c.values()),
            "native_coulomb_bound": all(m["native_coulomb_bound_passed"] for m in c.values()),
            "native_slide_friction_participation": all(
                c[k]["native_contact"]["free_friction_events"] > 0
                for k in ("slide_positive", "slide_negative", "oblique_landing")
            ),
            "native_friction_work_nonpositive": all(
                m["native_contact"]["positive_friction_work_j"] < 1e-10 for m in c.values()
            ),
            "native_no_excluded_friction": all(
                m["native_contact"]["attached_friction_events"] == 0
                and m["native_contact"]["noncontact_friction_events"] == 0
                for m in c.values()
            ),
            "normal_energy": all(m["normal_energy_passed"] for m in c.values()),
            "unforced_total_energy": all(
                m["total_energy_passed"] for k, m in c.items() if k != "attached"
            ),
            "supported_rest": c["rest"]["final_rms_speed_m_s"] < 0.005,
            "sliding_mu_g": all(
                c[k]["deceleration_relative_error"] is not None
                and c[k]["deceleration_relative_error"] < 0.15
                for k in ("slide_positive", "slide_negative")
            ),
            "reverse_symmetry": abs(
                c["slide_positive"]["stopping_distance_m"]
                - c["slide_negative"]["stopping_distance_m"]
            )
            < 0.002,
            "no_attached_friction": abs(c["attached"]["final_rms_speed_m_s"] - 0.5) < 0.005,
            "no_airborne_friction": c["airborne"]["max_horizontal_velocity_change_m_s"] < 0.005,
            "native_airborne_zero_impulses": (
                c["airborne"]["native_contact"]["tangential_impulse_kg_m_s"] == 0
                and c["airborne"]["native_contact"]["normal_impulse_kg_m_s"] == 0
            ),
            "native_attached_zero_friction": (
                c["attached"]["native_contact"]["tangential_impulse_kg_m_s"] == 0
            ),
        }
        if include_half_dt:
            d = c["slide_positive"]["stopping_distance_m"]
            gates["half_dt_stopping_distance"] = (
                abs(c["slide_positive_half_dt"]["stopping_distance_m"] - d) / max(d, 1e-9) < 0.10
            )
        report["gates"] = gates
        report["qualification_passed"] = all(gates.values())
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    report = run_contact_diagnostic(
        json.loads(Path(args.config).read_text()), output_dir=args.output_dir, backend=args.backend
    )
    print(json.dumps(report, indent=2))
    return 0 if report["qualification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
