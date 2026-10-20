"""Progressive diagonal deposition and a bounded slack-taking tug."""

import time
from dataclasses import asdict, dataclass

import numpy as np

from cloth_fling.contracts import ActionOutcome
from cloth_fling.reference_geometry import material_line, sample_line
from cloth_fling.reference_leading_motion import LeadingMotion
from cloth_fling.reference_leading_motion import acceptance as leading_acceptance
from cloth_fling.reference_motion import ReferenceMotionFailure
from cloth_fling.reference_trajectory import Quintic


@dataclass(frozen=True)
class DiagonalTrajectory:
    version: str = "reference_diagonal_v3"
    endpoint_x_m: float = 0.78
    diagonal_seconds: float = 6.0
    endpoint_hold_seconds: float = 1.0
    max_tug_m: float = 0.02

    def __post_init__(self):
        if self.version in ("reference_diagonal_drag_v4", "reference_diagonal_centered_drag_v5"):
            if (
                self.endpoint_x_m,
                self.diagonal_seconds,
                self.endpoint_hold_seconds,
                self.max_tug_m,
            ) != (0.73, 6.0, 1.0, 0.12):
                raise ValueError("trajectory differs from the authorized 12 cm final drag")
            return
        if (
            self.version != "reference_diagonal_v3"
            or self.endpoint_x_m not in (0.78, 0.76, 0.74)
            or self.diagonal_seconds not in (6, 8, 10, 4)
            or self.endpoint_hold_seconds != 1
            or self.max_tug_m != 0.02
        ):
            raise ValueError("trajectory differs from frozen diagonal grid")


def diagonal_grid():
    return [
        DiagonalTrajectory(endpoint_x_m=x, diagonal_seconds=t)
        for x in (0.78, 0.76, 0.74)
        for t in (6, 8, 10, 4)
    ]


def diagonal_schedule(config, dt=0.02):
    duration = config.diagonal_seconds
    while True:
        segment = Quintic.between(
            "diagonal",
            duration,
            [0.10, 0, 0.95],
            [config.endpoint_x_m, 0, 0.31],
            [0, 0, 0],
            [0, 0, 0],
        )
        factor = max(segment.maximum_norm(1) / 0.8, np.sqrt(segment.maximum_norm(2) / 2))
        if factor <= 1 + 1e-10:
            break
        duration = np.ceil(duration * factor / dt) * dt
    segments = [segment]
    if config.version in ("reference_diagonal_drag_v4", "reference_diagonal_centered_drag_v5"):
        drag = Quintic.between(
            "tug",
            2.0,
            [config.endpoint_x_m, 0, 0.31],
            [config.endpoint_x_m + config.max_tug_m, 0, 0.31],
            [0, 0, 0],
            [0, 0, 0],
        )
        if drag.maximum_norm(1) > 0.8 or drag.maximum_norm(2) > 2:
            raise ValueError("final drag exceeds kinematic bounds")
        segments.append(drag)
    fixed = 0.2 + 2 + 3 + duration + 1 + 2 + 1 + 0.2 + 5
    allowance = min(20.0, np.floor((35 - fixed + 1e-9) / dt) * dt)
    if allowance < 0:
        raise ValueError("complete diagonal schedule exceeds action budget")
    return segments, {
        "tauten_allowance_seconds": allowance,
        "maximum_action_seconds": fixed + allowance,
        "reserved_after_tauten_seconds": fixed - 2.2,
        "peak_speed_m_s": max(s.maximum_norm(1) for s in segments),
        "peak_acceleration_m_s2": max(s.maximum_norm(2) for s in segments),
        "segments": [
            {
                "phase": s.name,
                "seconds": s.seconds,
                "coefficients": s.coefficients.tolist(),
            }
            for s in segments
        ],
    }


def centered_placement_segments(env, target):
    """Development geometry estimate after the hold; never relocate free particles."""
    p = env.get_positions()
    midpoint = target.mean(axis=0)
    hanging_drop = max(0.0, float(midpoint[2] - p[:, 2].min()))
    body_y = float((p[:, 1].min() + p[:, 1].max()) / 2)
    center_x, center_y = env.config.table_center_xy_m
    # One measured development correction: folded cloth's trailing offset is
    # shorter than half its rest length. This is not a general unfolding model.
    trailing_offset = 0.43 * hanging_drop
    final_x = center_x + trailing_offset
    start_x = final_x - 0.12
    segments = [
        Quintic.between(
            "diagonal",
            6.0,
            [midpoint[0], 0, midpoint[2]],
            [start_x, center_y - body_y, 0.31],
            [0, 0, 0],
            [0, 0, 0],
        ),
        Quintic.between("tug", 2.0, [start_x, 0, 0.31], [final_x, 0, 0.31], [0, 0, 0], [0, 0, 0]),
    ]
    if any(s.maximum_norm(1) > 0.8 or s.maximum_norm(2) > 2 for s in segments):
        raise ReferenceMotionFailure("centered_placement_kinematic_limit")
    xmin, xmax, _, _ = env.config.table_bounds_m
    if not xmin <= start_x < final_x <= xmax:
        raise ReferenceMotionFailure("centering_target_outside_table")
    return segments, {
        "method": "hanging_drop_and_projected_Y_center_v1",
        "target_footprint_center_xy_m": [center_x, center_y],
        "measured_hanging_drop_m": hanging_drop,
        "measured_projected_Y_center_m": body_y,
        "trailing_drop_factor": 0.43,
        "estimated_trailing_offset_m": trailing_offset,
        "planned_pre_drag_midpoint_x_m": start_x,
        "planned_post_drag_midpoint_x_m": final_x,
        "planned_laydown_Y_displacement_m": center_y - body_y,
    }


class DepositedBody:
    """Frozen material IDs and landing references; membership cannot hide later slip."""

    def __init__(self, n):
        self.stable = np.zeros(n, int)
        self.landed = np.zeros(n, bool)
        self.cohorts = []
        self.cohort_id = np.full(n, -1, dtype=int)
        self.anchors = np.zeros((n, 2))
        self.main_max_p95 = 0.0
        self.at_height = None
        self.tug_ids = np.array([], int)
        self.tug_anchors = None
        self.tug_max_median = 0.0
        self.tug_max_p95 = 0.0

    def update(self, p, v, free, contacts, midpoint, phase):
        eligible = (
            free & contacts & (np.linalg.norm(v, axis=1) <= 0.01) & (p[:, 0] <= midpoint[0] - 0.10)
        )
        self.stable = np.where(eligible, self.stable + 1, 0)
        if phase == "diagonal":
            newly = (self.stable >= 5) & ~self.landed
            self.anchors[newly] = p[newly, :2]
            if newly.any():
                self.cohort_id[newly] = len(self.cohorts)
                self.cohorts.append(np.flatnonzero(newly))
            self.landed |= newly
            for ids in self.cohorts:
                self.main_max_p95 = max(
                    self.main_max_p95,
                    float(
                        np.quantile(np.linalg.norm(p[ids, :2] - self.anchors[ids], axis=1), 0.95)
                    ),
                )
            if midpoint[2] <= 0.4 and self.at_height is None:
                self.at_height = float(self.landed.sum() / free.sum())
        median, p95 = 0.0, 0.0
        if len(self.tug_ids):
            delta = np.linalg.norm(p[self.tug_ids, :2] - self.tug_anchors, axis=1)
            median, p95 = float(np.median(delta)), float(np.quantile(delta, 0.95))
            if phase in ("tug", "endpoint_hold"):
                self.tug_max_median = max(self.tug_max_median, median)
                self.tug_max_p95 = max(self.tug_max_p95, p95)
        return {
            "landed_fraction": float(self.landed.sum() / free.sum()),
            "main_max_drift_p95_m": self.main_max_p95,
            "tug_drift_median_m": median,
            "tug_drift_p95_m": p95,
        }

    def freeze_tug(self, p, free, *, require_body=True):
        self.tug_ids = np.flatnonzero((self.stable >= 5) & free)
        if require_body and len(self.tug_ids) < 0.25 * free.sum():
            raise ReferenceMotionFailure("insufficient_deposited_body")
        self.tug_anchors = p[self.tug_ids, :2].copy()

    def metrics(self, n_free):
        return {
            "deposited_before_height_fraction": self.at_height,
            "main_max_drift_p95_m": self.main_max_p95,
            "tug_body_fraction": len(self.tug_ids) / n_free,
            "tug_max_drift_median_m": self.tug_max_median,
            "tug_max_drift_p95_m": self.tug_max_p95,
        }


class DiagonalMotion(LeadingMotion):
    def __init__(self, env, *, execution_mode="qualification"):
        if execution_mode not in ("qualification", "demonstration"):
            raise ValueError("unknown diagonal execution mode")
        self.execution_mode = execution_mode
        self.body = DepositedBody(len(env.rest_positions))
        self.particle_contacts = []
        self.slack = []
        self.slack_stable = 0
        self.lines = []
        self.tug_stop_reason = None
        self.tug_distance = 0.0
        super().__init__(env, stop_on_placement_failure=execution_mode == "qualification")

    def sample(self, phase):
        p, v, rms = super().sample(phase)
        contact = self.env.scene.sim.coupler.take_particle_contact(reset=True)
        self.particle_contacts.append(
            {"simulated_seconds": self.env.simulated_time - self.sim_started, **contact}
        )
        contacts = contact["values"][:, 0] > 0
        if phase == "airborne_hold" and contacts.any():
            self.placement_failure("airborne_contact_regression")
        if self.ids:
            metrics = self.body.update(
                p, v, self.env._free(), contacts, self.target.mean(axis=0), phase
            )
            self.commands[-1].update(metrics)
            if (
                phase == "diagonal"
                and self.body.at_height is not None
                and self.body.at_height < 0.25
            ):
                self.placement_failure("insufficient_progressive_deposition")
            if phase == "diagonal" and self.body.main_max_p95 > 0.01:
                self.placement_failure("deposited_body_sliding")
            if (
                phase in ("diagonal", "tug", "endpoint_hold")
                and self.edge_crossings[-1]["crossed_vertices"].size
            ):
                self.placement_failure("placement_far_edge_crossing")
        if self.lines:
            ratios = []
            for indices, weights in self.lines:
                points = sample_line(p, indices, weights)
                arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                ratios.append(float(np.linalg.norm(points[-1] - points[0]) / max(arc, 1e-12)))
            self.slack_stable = self.slack_stable + 1 if min(ratios) >= 0.98 else 0
            self.slack.append(
                {
                    "simulated_seconds": self.env.simulated_time - self.sim_started,
                    "chord_arclength_ratios": ratios,
                    "consecutive_taut_samples": self.slack_stable,
                }
            )
        return p, v, rms

    def prepare_final_contact(self):
        rest = self.env.rest_positions
        for vertex in self.ids:
            target = [rest[:, 0].max() - 0.015, rest[vertex, 1]]
            far = int(np.argmin(np.linalg.norm(rest[:, :2] - target, axis=1)))
            self.lines.append(material_line(rest, self.env.faces, rest[vertex], rest[far]))
        self.hold("pre_tug_stabilize", 1.0)
        self.body.freeze_tug(
            self.env.get_positions(),
            self.env._free(),
            require_body=self.stop_on_placement_failure,
        )
        if len(self.body.tug_ids) < 0.25 * self.env._free().sum():
            self.placement_failure("insufficient_deposited_body")

    def perform_drag(self, segment):
        """Full continuous forward drag; supported-body motion is expected here."""
        self.prepare_final_contact()
        start_x = float(self.target[:, 0].mean())
        self.tug_stop_reason = "continuous_drag"
        try:
            self.curve(segment)
            self.tug_stop_reason = "completed_full_drag"
        finally:
            self.tug_distance = float(self.target[:, 0].mean() - start_x)

    def perform_tug(self):
        self.prepare_final_contact()
        self.tug_stop_reason = "distance_cap"
        for _ in range(20):
            if self.slack_stable >= 5:
                self.tug_stop_reason = "geometric_slack_removed"
                break
            p = self.env.get_positions()
            if len(self.body.tug_ids):
                delta = np.linalg.norm(p[self.body.tug_ids, :2] - self.body.tug_anchors, axis=1)
                if np.median(delta) >= 0.001 or np.quantile(delta, 0.95) >= 0.003:
                    self.tug_stop_reason = "incipient_body_slip"
                    break
            start = self.target.mean(axis=0)
            # curve() adds the held midpoint Y; segments use midpoint-relative Y.
            # A world-Y start would add that offset twice at the first substep.
            start[1] = 0.0
            self.curve(
                Quintic.between("tug", 0.1, start, start + [0.001, 0, 0], [0, 0, 0], [0, 0, 0])
            )
            self.tug_distance += 0.001


def acceptance(outcome):
    result = leading_acceptance(outcome)
    d = outcome.diagnostics
    result.update(
        retained_pins=bool(
            d.get("tracking_error_m", 1) <= 1e-6
            and all(
                len(c["attachment_ids"]) == 2
                for c in outcome.evidence["commands"]
                if c["phase"] in ("diagonal", "tug", "endpoint_hold")
            )
        ),
        progressive_deposition=d.get("deposited_before_height_fraction") is not None
        and d["deposited_before_height_fraction"] >= 0.25,
        landed_body_stays=d.get("main_max_drift_p95_m", 1) <= 0.01,
        tug_body_present=d.get("tug_body_fraction", 0) >= 0.25,
        tug_body_stays=d.get("tug_max_drift_median_m", 1) <= 0.002
        and d.get("tug_max_drift_p95_m", 1) <= 0.005,
    )
    return result


def campaign_stopping(outcome):
    return (
        outcome.reason
        not in {
            "completed",
            "placement_far_edge_crossing",
            "airborne_clearance_failure",
            "release_not_near_rest",
            "settling_timeout",
            "insufficient_progressive_deposition",
            "deposited_body_sliding",
            "insufficient_deposited_body",
        }
        or not outcome.evidence["cleanup"]["passed"]
    )


def execute_diagonal_action(
    env,
    action,
    trajectory,
    *,
    expected_ids,
    cleanup_callback,
    execution_mode="qualification",
    retain_existing_grasp=False,
):
    if retain_existing_grasp:
        ids = tuple(env.attached_ids)
        free = np.asarray(env._free(), dtype=bool)
        if (
            len(ids) != 2
            or len(set(ids)) != 2
            or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in ids)
            or ids != tuple(expected_ids)
            or set(ids) != set(np.flatnonzero(~free))
        ):
            # The caller still owns an invalid handoff; do not release or mutate it.
            raise ValueError("handoff requires exactly the expected two existing native pins")
        if env.get_positions()[ids[0], 1] > env.get_positions()[ids[1], 1]:
            raise ValueError("handoff native pins must be ordered from negative to positive Y")
    segments, schedule = diagonal_schedule(trajectory, env.config.controller_dt)
    before = env.observe()
    coverage_before, area_before, flat_area = env.coverage()
    run = DiagonalMotion(env, execution_mode=execution_mode)
    reason, failure, settled, cleanup = "completed", None, False, {}
    terminal_state, after = None, None
    release_ordering = []
    placement_centering = None
    try:
        run.ids = tuple(env.attached_ids) if retain_existing_grasp else env.attach(action.pixels)
        if run.ids != tuple(expected_ids):
            raise ReferenceMotionFailure("favorable_vertex_identity_mismatch")
        run.target = env.get_positions()[list(run.ids)].copy()
        run.hold("grasp_handoff" if retain_existing_grasp else "attach", 0.2)
        midpoint = run.target.mean(axis=0)
        span = float(np.linalg.norm((run.target[1] - run.target[0])[:2]))
        goal = np.array(
            [[0.10, midpoint[1] - span / 2, 0.95], [0.10, midpoint[1] + span / 2, 0.95]]
        )
        run.move("lift", goal, 2.0)
        run.tauten_cloth(schedule["tauten_allowance_seconds"])
        run.hold("airborne_hold", 3.0)
        if run.hold_clearance <= 0:
            run.placement_failure("airborne_clearance_failure")
        if trajectory.version == "reference_diagonal_centered_drag_v5":
            segments, placement_centering = centered_placement_segments(env, run.target)
            schedule["segments"] = [
                {"phase": s.name, "seconds": s.seconds, "coefficients": s.coefficients.tolist()}
                for s in segments
            ]
            schedule["peak_speed_m_s"] = max(s.maximum_norm(1) for s in segments)
            schedule["peak_acceleration_m_s2"] = max(s.maximum_norm(2) for s in segments)
        if trajectory.version in (
            "reference_diagonal_drag_v4",
            "reference_diagonal_centered_drag_v5",
        ):
            run.curve(segments[0])
            run.perform_drag(segments[1])
        else:
            for segment in segments:
                run.curve(segment)
            run.perform_tug()
        run.hold("endpoint_hold", trajectory.endpoint_hold_seconds)
        release_sec = env.simulated_time - run.sim_started
        release_ordering = [
            e for e in run.ordering if e["simulated_seconds"] >= release_sec - 0.2 - 1e-8
        ]
        run.release_tcp_speed = float(
            np.linalg.norm(env.get_velocities()[list(run.ids)], axis=1).max()
        )
        if run.release_tcp_speed > 0.01:
            run.placement_failure("release_not_near_rest")
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
            run.placement_failure("settling_timeout")
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
        and (settled or execution_mode == "demonstration")
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
            "execution_mode": execution_mode,
            "settled": settled,
            **run.body.metrics(len(env.rest_positions) - 2),
            "tug_stop_reason": run.tug_stop_reason,
            "tug_distance_m": run.tug_distance,
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
            "retained_existing_grasp": retain_existing_grasp,
            "placement_violations": run.placement_violations,
            "cleanup": cleanup,
            "failure": failure,
            "phases": run.phases,
            "config": {"profile": "reference_diagonal_v3", "table_height_m": 0.3},
            "trajectory": asdict(trajectory),
            "schedule": schedule,
            "placement_centering": placement_centering,
            "commands": run.commands,
            "particle_contacts": run.particle_contacts,
            "slack_traces": run.slack,
            "tug_body_ids": run.body.tug_ids,
            "tug_body_anchors_xy_m": run.body.tug_anchors,
            "deposited_ids": np.flatnonzero(run.body.landed),
            "deposited_cohort_ids": run.body.cohort_id,
            "deposited_anchors_xy_m": run.body.anchors[run.body.landed],
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
