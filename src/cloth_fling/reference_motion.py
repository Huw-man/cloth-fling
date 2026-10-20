"""Prescribed retained-grasp laydown; only the grasp choice is policy-controlled."""

import time
from dataclasses import dataclass

import numpy as np

from cloth_fling.contracts import ActionOutcome
from cloth_fling.reference_geometry import material_line, sample_line


@dataclass(frozen=True)
class ReferenceMotionConfig:
    profile: str = "reference_laydown_v1"
    table_height_m: float = 0.30
    max_simulated_seconds: float = 35.0
    max_wall_seconds: float = 180.0
    settle_rms_m_s: float = 0.01
    settle_timeout_seconds: float = 5.0
    gross_edge_ratio: float = 3.0

    def __post_init__(self):
        if (
            self.profile != "reference_laydown_v1"
            or self.table_height_m != 0.30
            or self.gross_edge_ratio != 3.0
            or not 0 < self.max_simulated_seconds <= 35
            or not 0 < self.max_wall_seconds <= 180
            or not 0 < self.settle_rms_m_s <= 0.01
            or not 0 < self.settle_timeout_seconds <= 5
        ):
            raise ValueError("reference motion exceeds frozen limits")


def minimum_jerk(u):
    return 10 * u**3 - 15 * u**4 + 6 * u**5


class ReferenceMotionFailure(RuntimeError):
    pass


class _Motion:
    def __init__(self, env, config, callback):
        self.env, self.config, self.callback = env, config, callback
        self.started, self.sim_started = time.monotonic(), env.simulated_time
        self.tick_dt = env.config.controller_dt
        self.subticks = round(self.tick_dt / env.config.sim_dt)
        self.target = None
        self.ids = ()
        self.commands, self.trace, self.phases = [], [], []
        self.max_ratio = 1.0
        self.hold_clearance = float("inf")
        self.hold_samples = 0
        self.tauten = {}
        self.release_tcp_speed = None
        self.last_target = None
        self.terminal_valid = False

    def deadline(self):
        if time.monotonic() - self.started >= self.config.max_wall_seconds:
            raise ReferenceMotionFailure("reference_wall_deadline")
        if self.env.simulated_time - self.sim_started > self.config.max_simulated_seconds + 1e-8:
            raise ReferenceMotionFailure("reference_simulated_deadline")

    def sample(self, phase):
        self.deadline()
        p, v = self.env.get_positions(), self.env.get_velocities()
        if not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ReferenceMotionFailure("nonfinite_state")
        if (p < [-1.2, -1.2, -0.1]).any() or (p > [1.2, 1.2, 2.0]).any():
            raise ReferenceMotionFailure("solver_domain_escape")
        ratio = float(
            (
                np.linalg.norm(p[self.env.edges[:, 1]] - p[self.env.edges[:, 0]], axis=1)
                / self.env.rest_edge_lengths
            ).max()
        )
        self.max_ratio = max(self.max_ratio, ratio)
        if ratio >= self.config.gross_edge_ratio:
            raise ReferenceMotionFailure("gross_edge_ratio")
        free = self.env._free()
        if self.ids and (
            tuple(self.env.attached_ids) != self.ids
            or free[list(self.ids)].any()
            or np.count_nonzero(~free) != 2
        ):
            raise ReferenceMotionFailure("native_ownership_loss")
        rms = float(np.sqrt(np.mean(np.sum(v**2, axis=1))))
        clearance = float(
            p[:, 2].min() - self.env.config.table_surface_z_m - self.env.particle_diameter / 2
        )
        if phase == "airborne_hold":
            self.hold_clearance = min(self.hold_clearance, clearance)
            self.hold_samples += 1
        entry = {
            "phase": phase,
            "simulated_seconds": self.env.simulated_time - self.sim_started,
            "commanded_grasp_positions_m": None if self.target is None else self.target.copy(),
            "attachment_ids": list(self.env.attached_ids),
            "rms_speed_m_s": rms,
            "min_surface_clearance_m": clearance,
            "max_edge_ratio": ratio,
        }
        self.commands.append(entry)
        if len(self.commands) % 5 == 0:
            trace = {
                **entry,
                "cloth_positions_m": p.copy(),
                "cloth_velocities_m_s": v.copy(),
                "grasp_particle_ids": list(self.ids),
                "measured_grasp_positions_m": p[list(self.ids)].copy() if self.ids else [],
                "contact_accounting": self.env.take_contact_accounting(),
            }
            self.trace.append(trace)
            if self.callback:
                self.callback(self.env, trace)
            self.deadline()
        return p, v, rms

    def tick(self, phase, target=None):
        start = None if self.target is None else self.target.copy()
        if target is not None and start is None:
            start = np.asarray(target).copy()
        for step in range(1, self.subticks + 1):
            self.deadline()
            if target is not None:
                waypoint = start + step / self.subticks * (target - start)
                self.env.set_attachment_targets(waypoint)
            self.env.step()
        if target is not None:
            self.last_target = self.target
            self.target = np.asarray(target).copy()
        return self.sample(phase)

    def phase(self, name):
        self.phases.append(
            {"phase": name, "start_seconds": self.env.simulated_time - self.sim_started}
        )

    def hold(self, name, seconds):
        self.phase(name)
        result = None
        for _ in range(round(seconds / self.tick_dt)):
            result = self.tick(name, self.target if self.ids else None)
        return result

    def move(self, name, goal, seconds):
        self.phase(name)
        start = self.target.copy()
        count = int(np.ceil(seconds / self.tick_dt - 1e-10))
        for index in range(1, count + 1):
            self.tick(name, start + minimum_jerk(index / count) * (goal - start))

    def tauten_cloth(self):
        self.phase("tauten")
        rest = self.env.rest_positions
        # Face centers define material samples; grasp vertices remain independent hard pins.
        centers = []
        for vertex in self.ids:
            face = self.env.faces[np.flatnonzero((self.env.faces == vertex).any(axis=1))[0]]
            centers.append(rest[face].mean(axis=0))
        indices, weights = material_line(rest, self.env.faces, *centers)
        line = sample_line(rest, indices, weights)
        rest_length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        midpoint = self.target.mean(axis=0)
        initial = float(np.linalg.norm(self.target[1] - self.target[0]))
        cap = max(initial, min(0.70, 1.02 * rest_length))
        begin = self.env.simulated_time
        stable, reason = 0, "duration_bound"

        def feedback(p, rms):
            nonlocal stable
            points = sample_line(p, indices, weights)
            chord = float(np.linalg.norm(points[-1] - points[0]))
            baseline = np.linspace(points[0], points[-1], len(points))
            sag = float(np.maximum(baseline[:, 2] - points[:, 2], 0).max())
            stable = (
                stable + 1 if chord / rest_length >= 0.95 and sag <= 0.02 and rms <= 0.01 else 0
            )
            self.tauten.update(
                span_rest_ratio=chord / rest_length, downward_sag_m=sag, consecutive_samples=stable
            )
            return stable >= 5

        span = initial
        while self.env.simulated_time - begin < 20 - self.tick_dt / 2:
            p, _, rms = self.tick("tauten", self.target)
            if feedback(p, rms):
                reason = "tautness_proxy"
                break
            if span >= cap - 1e-9:
                reason = "span_bound"
                break
            delta = min(0.02, cap - span)
            duration = np.ceil(max(0.4, 1.875 * (delta / 2) / 0.02) / self.tick_dt) * self.tick_dt
            if self.env.simulated_time - begin + duration > 20 + 1e-8:
                break
            old = self.target.copy()
            span += delta
            goal = midpoint + np.array([[0, -span / 2, 0], [0, span / 2, 0]])
            count = round(duration / self.tick_dt)
            done = False
            for i in range(1, count + 1):
                p, _, rms = self.tick("tauten", old + minimum_jerk(i / count) * (goal - old))
                feedback(p, rms)
            for _ in range(
                min(50, int((20 - (self.env.simulated_time - begin) + 1e-8) / self.tick_dt))
            ):
                p, _, rms = self.tick("tauten", self.target)
                if feedback(p, rms):
                    reason, done = "tautness_proxy", True
                    break
            if done:
                break
        self.tauten.update(
            stop_reason=reason,
            rest_line_length_m=rest_length,
            cap_m=cap,
            final_span_m=span,
            material_indices=indices,
            barycentric_weights=weights,
        )


def execute_reference_action(
    env, action, *, config=None, trace_callback=None, cleanup_callback=None
):
    config = config or ReferenceMotionConfig()
    before = env.observe()
    coverage_before, area_before, flat_area = env.coverage()
    run = _Motion(env, config, trace_callback)
    reason, settled, after = "completed", False, None
    failure, cleanup = None, {}
    selection = {}
    terminal_state = None
    try:
        if action.primitive != "laydown":
            raise ValueError("reference profile executes laydown only")
        if getattr(action, "encoding", None):
            from cloth_fling.spatial_actions import build_action_batch

            batch = build_action_batch(before, env.config)
            index = tuple(action.encoding["map_index_tvu"])
            if (
                action.encoding["observation_id"] != before.observation_id
                or not batch.feasible[index]
                or not np.allclose(batch.decode(index[0], index[2], index[1]), action.pixels)
            ):
                raise ValueError("spatial action identity mismatch")
            selection = {
                "encoding": action.encoding,
                "valid_action_mask": batch.feasible,
                "to_original": batch.to_original,
                "crop": batch.crop,
            }
        run.ids = env.attach(action.pixels)
        run.target = env.get_positions()[list(run.ids)].copy()
        run.hold("attach", 0.2)
        midpoint = run.target.mean(axis=0)
        span = float(np.linalg.norm((run.target[1] - run.target[0])[:2]))
        goal = np.array(
            [[0.10, midpoint[1] - span / 2, 0.95], [0.10, midpoint[1] + span / 2, 0.95]]
        )
        run.move("lift", goal, 2.0)
        run.tauten_cloth()
        run.hold("airborne_hold", 3.0)
        goal = run.target + np.array([0.70, midpoint[1], 0.31]) - run.target.mean(axis=0)
        run.move("laydown", goal, 2.0)
        run.move("drag", run.target + [0.10, 0, 0], 0.8)
        run.release_tcp_speed = float(
            np.linalg.norm((run.target - run.last_target) / run.tick_dt, axis=1).max()
        )
        env.release()
        run.ids = ()
        run.target = None
        run.hold("release", 0.2)
        run.phase("settle")
        stable = 0
        for _ in range(round(config.settle_timeout_seconds / run.tick_dt)):
            _, _, rms = run.tick("settle")
            stable = stable + 1 if rms <= config.settle_rms_m_s else 0
            if stable >= 5:
                settled = True
                break
        if not settled:
            raise ReferenceMotionFailure("settling_timeout")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        reason = str(exc)
    finally:
        # This callback persists independently of report or transition writing.
        try:
            env.release()
            cleanup = {
                "released": not env.attached_ids,
                "native_free": bool(env._free().all()),
                "release": env.release_evidence,
            }
            cleanup["passed"] = cleanup["released"] and cleanup["native_free"]
        except Exception as exc:
            cleanup = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        if cleanup_callback:
            cleanup_callback(cleanup)
    if not cleanup["passed"]:
        reason = "cleanup_failed"
    try:
        after = env.observe()
        coverage_after, area_after, _ = env.coverage()
        terminal_state = env.snapshot()
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
    except Exception as exc:
        reason = f"terminal_artifact_failure: {exc}"
        after, coverage_after, area_after, containment = None, float("nan"), float("nan"), None
    valid = bool(failure is None and cleanup["passed"] and after is not None and settled)
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
        simulated_seconds=env.simulated_time - run.sim_started,
        wall_seconds=time.monotonic() - run.started,
        terminated=not valid,
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
        },
        evidence={
            "cleanup": cleanup,
            "failure": failure,
            "phases": run.phases,
            "config": {"profile": config.profile, "table_height_m": config.table_height_m},
            "commands": run.commands,
            "phase_traces": run.trace,
            "selection": selection,
            "terminal_state": terminal_state,
            "native_faces": env.faces.copy(),
            "native_rest_positions": env.rest_positions.copy(),
            "contact": env.take_contact_accounting(),
        },
    )
