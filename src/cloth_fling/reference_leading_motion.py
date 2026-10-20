"""Native execution and evidence for the separately authorized leading-edge campaign."""

import time
from dataclasses import asdict

import numpy as np

from cloth_fling.contracts import ActionOutcome
from cloth_fling.reference_motion import ReferenceMotionConfig, ReferenceMotionFailure, _Motion
from cloth_fling.reference_trajectory import leading_fraction, placement_schedule


class LeadingMotion(_Motion):
    def __init__(self, env, callback=None):
        super().__init__(env, ReferenceMotionConfig(), callback)
        self.native_states = []
        self.ordering = []
        self.edge_crossings = []
        self.native_capture("time_zero")

    def native_capture(self, name):
        state = self.env.snapshot()
        p, v = state["positions"], state["velocities"]
        entry = {
            "phase": name,
            "simulated_seconds": self.env.simulated_time - self.sim_started,
            "cloth_positions_m": p.copy(),
            "cloth_velocities_m_s": v.copy(),
            "attachment_ids": list(self.env.attached_ids),
            "commanded_grasp_positions_m": None if self.target is None else self.target.copy(),
            "measured_grasp_positions_m": p[list(self.ids)].copy() if self.ids else [],
            "measured_grasp_velocities_m_s": v[list(self.ids)].copy() if self.ids else [],
            "coverage": self.env.coverage()[0] if np.isfinite(p).all() else None,
            "contact_accounting": self.env.take_contact_accounting(),
        }
        self.native_states.append({"event": name, "state": state})
        self.trace.append(entry)

    def phase(self, name):
        self.native_capture("boundary_" + name)
        super().phase(name)

    def sample(self, phase):
        p, v, rms = super().sample(phase)
        free = self.env._free()
        sec = self.env.simulated_time - self.sim_started
        fraction = leading_fraction(p, free, self.target[:, 0].mean()) if self.ids else None
        crossing = (p[:, 0] > self.env.config.table_bounds_m[1]) & (p[:, 2] < 0.35) & free
        entry = {
            "simulated_seconds": sec,
            "phase": phase,
            "crossed_vertices": np.flatnonzero(crossing),
            "free_max_x_m": float(p[free, 0].max()),
            "leading_fraction": fraction,
        }
        self.edge_crossings.append(entry)
        if self.ids:
            self.ordering.append({"simulated_seconds": sec, "fraction": fraction})
        self.commands[-1].update(
            leading_fraction=fraction,
            edge_crossing_count=int(crossing.sum()),
            measured_grasp_velocities_m_s=v[list(self.ids)].copy() if self.ids else [],
        )
        if self.trace and abs(self.trace[-1]["simulated_seconds"] - sec) < 1e-8:
            self.trace[-1].update(
                leading_fraction=fraction, edge_crossing_count=int(crossing.sum())
            )
        if phase in ("transfer", "descent", "drag", "endpoint_hold") and crossing.any():
            raise ReferenceMotionFailure("placement_far_edge_crossing")
        return p, v, rms

    def curve(self, segment):
        self.phase(segment.name)
        offset = self.target - self.target.mean(axis=0)
        offset[:, 1] += self.target[:, 1].mean()
        for tick in range(round(segment.seconds / self.tick_dt)):
            previous = self.target.copy()
            for step in range(1, self.subticks + 1):
                self.deadline()
                t = tick * self.tick_dt + step * self.env.config.sim_dt
                target = segment.at(min(t, segment.seconds)) + offset
                self.env.set_attachment_targets(target)
                self.env.step()
            self.last_target, self.target = previous, target
            self.sample(segment.name)


def execute_leading_action(env, action, trajectory, *, expected_ids, cleanup_callback):
    segments, schedule = placement_schedule(trajectory, env.config.controller_dt)
    before = env.observe()
    coverage_before, area_before, flat_area = env.coverage()
    run = LeadingMotion(env)
    reason, failure, settled, cleanup = "completed", None, False, {}
    terminal_state, after = None, None
    release_ordering = []
    try:
        run.ids = env.attach(action.pixels)
        if run.ids != tuple(expected_ids):
            raise ReferenceMotionFailure("favorable_vertex_identity_mismatch")
        run.target = env.get_positions()[list(run.ids)].copy()
        run.hold("attach", 0.2)
        midpoint = run.target.mean(axis=0)
        span = float(np.linalg.norm((run.target[1] - run.target[0])[:2]))
        goal = np.array(
            [[0.10, midpoint[1] - span / 2, 0.95], [0.10, midpoint[1] + span / 2, 0.95]]
        )
        run.move("lift", goal, 2.0)
        run.tauten_cloth(schedule["tauten_allowance_seconds"])
        run.hold("airborne_hold", 3.0)
        if run.hold_clearance <= 0:
            raise ReferenceMotionFailure("airborne_clearance_failure")
        for segment in segments:
            run.curve(segment)
        run.hold("endpoint_hold", trajectory.endpoint_hold_seconds)
        release_sec = env.simulated_time - run.sim_started
        release_ordering = [
            e for e in run.ordering if e["simulated_seconds"] >= release_sec - 0.2 - 1e-8
        ]
        run.release_tcp_speed = float(
            np.linalg.norm(env.get_velocities()[list(run.ids)], axis=1).max()
        )
        if run.release_tcp_speed > 0.01:
            raise ReferenceMotionFailure("release_not_near_rest")
        run.native_capture("before_release")
        env.release()
        if (
            not env.release_evidence.get("passed")
            or env.release_evidence.get("velocity_error_m_s", 1) > 1e-4
        ):
            raise ReferenceMotionFailure("release_velocity_regression")
        run.ids, run.target = (), None
        run.hold("release", 0.2)
        run.phase("settle")
        stable = 0
        # Always permit the full five seconds of unsupported settling; evaluate the endpoint.
        for _ in range(round(5 / run.tick_dt)):
            _, _, rms = run.tick("settle")
            stable = stable + 1 if rms <= 0.01 else 0
        settled = stable >= 5
        if not settled:
            raise ReferenceMotionFailure("settling_timeout")
    except Exception as exc:
        reason, failure = str(exc), f"{type(exc).__name__}: {exc}"
        try:
            run.native_capture("failure")
        except Exception as artifact_exc:
            reason = f"failure_artifact_regression: {artifact_exc}"
    finally:
        # The certificate is durable before any report, replay or outcome serialization.
        try:
            env.release()
            cleanup = {
                "released": not env.attached_ids,
                "native_free": bool(env._free().all()),
                "release": env.release_evidence,
            }
            cleanup["passed"] = bool(
                cleanup["released"]
                and cleanup["native_free"]
                and cleanup["release"].get("passed")
                and cleanup["release"].get("velocity_error_m_s", 1) <= 1e-4
            )
        except Exception as exc:
            cleanup = {"passed": False, "error": str(exc)}
        cleanup_callback(cleanup)
    if not cleanup["passed"]:
        reason = "cleanup_failed"
    coverage_after, area_after, containment, terminal_rms = float("nan"), float("nan"), None, None
    try:
        run.ids, run.target = (), None
        run.native_capture("termination")
        terminal_state = env.snapshot()
        after = env.observe()
        coverage_after, area_after, _ = env.coverage()
        terminal = env.get_positions()
        xmin, xmax, ymin, ymax = env.config.table_bounds_m
        containment = float(
            (
                (terminal[:, 0] >= xmin)
                & (terminal[:, 0] <= xmax)
                & (terminal[:, 1] >= ymin)
                & (terminal[:, 1] <= ymax)
            ).mean()
        )
        terminal_rms = float(np.sqrt(np.mean(np.sum(env.get_velocities() ** 2, axis=1))))
    except Exception as exc:
        reason = f"terminal_artifact_failure: {exc}"
    valid = bool(
        failure is None
        and cleanup["passed"]
        and terminal_state is not None
        and after is not None
        and settled
    )
    return ActionOutcome(
        before=before,
        after=after,
        action=action,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        area_before_m2=area_before,
        area_after_m2=area_after,
        flat_area_m2=flat_area,
        execution_valid=valid,
        reason=reason,
        terminated=not valid,
        simulated_seconds=env.simulated_time - run.sim_started,
        wall_seconds=time.monotonic() - run.started,
        diagnostics={
            "settled": settled,
            "terminal_observation_valid": after is not None,
            "max_edge_ratio": run.max_ratio,
            "tauten": run.tauten,
            "tracking_error_m": env.tracking_error_m,
            "airborne_hold_seconds": run.hold_samples * run.tick_dt,
            "airborne_hold_min_clearance_m": run.hold_clearance if run.hold_samples else None,
            "release_tcp_speed_m_s": run.release_tcp_speed,
            "final_table_containment": containment,
            "terminal_rms_speed_m_s": terminal_rms,
            "leading_window_samples": len(release_ordering),
            "leading_window_seconds": release_ordering[-1]["simulated_seconds"]
            - release_ordering[0]["simulated_seconds"]
            if release_ordering
            else 0,
            "leading_window_min_fraction": min(
                (e["fraction"] for e in release_ordering), default=None
            ),
        },
        evidence={
            "cleanup": cleanup,
            "failure": failure,
            "phases": run.phases,
            "config": {"profile": "reference_laydown_v1", "table_height_m": 0.3},
            "trajectory": asdict(trajectory),
            "schedule": schedule,
            "commands": run.commands,
            "phase_traces": run.trace,
            "native_states": run.native_states,
            "leading_ordering": run.ordering,
            "edge_crossings": run.edge_crossings,
            "terminal_state": terminal_state,
            "native_faces": env.faces.copy(),
            "native_rest_positions": env.rest_positions.copy(),
            "contact": env.take_contact_accounting(),
        },
    )


def acceptance(outcome):
    d, release = outcome.diagnostics, outcome.evidence["cleanup"].get("release", {})
    return {
        "execution": bool(outcome.execution_valid),
        "airborne": bool(
            d.get("airborne_hold_seconds", 0) >= 3 - 1e-8
            and (d.get("airborne_hold_min_clearance_m") or -1) > 0
        ),
        "retained_pins": bool(
            d.get("tracking_error_m", 1) <= 1e-6
            and all(
                len(e["attachment_ids"]) == 2
                for e in outcome.evidence["commands"]
                if e["phase"] in ("transfer", "descent", "drag", "endpoint_hold")
            )
        ),
        "leading_edge": bool(
            d.get("leading_window_seconds", 0) >= 0.2 - 1e-8
            and (d.get("leading_window_min_fraction") or 0) >= 0.95
        ),
        "coverage": bool(outcome.coverage_after >= 0.95),
        "containment": d.get("final_table_containment") == 1.0,
        "settled": bool(d.get("settled")),
        "near_rest_release": d.get("release_tcp_speed_m_s") is not None
        and d["release_tcp_speed_m_s"] <= 0.01,
        "release_velocity": bool(
            release.get("passed") and release.get("velocity_error_m_s", 1) <= 1e-4
        ),
        "cleanup": bool(outcome.evidence["cleanup"]["passed"]),
    }


def campaign_stopping(outcome):
    ordinary = {
        "completed",
        "placement_far_edge_crossing",
        "airborne_clearance_failure",
        "release_not_near_rest",
        "settling_timeout",
    }
    return outcome.reason not in ordinary or not outcome.evidence["cleanup"]["passed"]
