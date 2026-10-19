"""Bounded attachment-assisted primitives; depends only on the backend protocol and NumPy."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cloth_fling.contracts import ActionOutcome, ClothBackend, GraspAction
from cloth_fling.selection import foreground_mask, pixels_to_world


@dataclass(frozen=True)
class PrimitiveConfig:
    """SI units. Shorter durations/settling windows support CPU and integrated smoke tests."""

    lift_height_m: float = 0.30
    table_height_m: float = 0.0
    release_height_m: float = 0.008
    lift_seconds: float = 1.8
    tauten_seconds: float = 2.0
    laydown_seconds: float = 2.0
    drag_seconds: float = 0.8
    fling_segment_seconds: float = 0.45
    laydown_distance_m: float = 0.12
    drag_distance_m: float = 0.06
    swing_distance_m: float = 0.16
    pullback_distance_m: float = 0.10
    place_distance_m: float = 0.04
    max_speed_m_s: float = 1.4
    max_acceleration_m_s2: float = 5.0
    tauten_speed_m_s: float = 0.08  # total span rate; each attachment moves half this
    max_strain: float = 0.10
    taut_rest_ratio: float = 0.98
    taut_sag_tolerance_m: float = 0.008
    fallback_spread_m: float = 0.08
    settle_rms_m_s: float = 0.01
    settle_hold_seconds: float = 0.5
    settle_timeout_seconds: float = 10.0
    max_simulated_seconds: float = 30.0
    max_wall_seconds: float = 120.0
    trace_interval_seconds: float = 0.05

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("primitive config must be finite")
        nonnegative = {
            "table_height_m",
            "laydown_distance_m",
            "drag_distance_m",
            "swing_distance_m",
            "pullback_distance_m",
            "place_distance_m",
            "fallback_spread_m",
            "max_strain",
        }
        if any(v < 0 if k in nonnegative else v <= 0 for k, v in values.items()):
            raise ValueError("primitive config contains invalid SI value")
        if not (
            self.max_simulated_seconds <= 30
            and self.max_wall_seconds <= 120
            and self.settle_timeout_seconds <= 10
            and self.settle_rms_m_s <= 0.01
            and self.settle_hold_seconds <= self.settle_timeout_seconds
            and self.release_height_m <= 0.02
            and self.release_height_m < self.lift_height_m
            and 0 < self.taut_rest_ratio <= 1
        ):
            raise ValueError("primitive config exceeds action/settle/release bounds")


class PrimitiveFailure(RuntimeError):
    pass


class DeadlineExceeded(PrimitiveFailure):
    pass


def _state(env):
    p = np.asarray(env.get_positions(), dtype=float)
    v = np.asarray(env.get_velocities(), dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or v.shape != p.shape or len(p) < 2:
        raise PrimitiveFailure("invalid_cloth_state_shape")
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise PrimitiveFailure("nonfinite_cloth_state")
    return p.copy(), v.copy()


def _coverage(env):
    result = tuple(float(x) for x in env.coverage())
    if len(result) != 3 or not np.isfinite(result).all() or result[2] <= 0:
        raise PrimitiveFailure("invalid_coverage")
    return result


@dataclass(frozen=True)
class _RestGeometry:
    positions: np.ndarray
    edges: np.ndarray
    lengths: np.ndarray


def _edge_strain(geometry, positions, velocities=None):
    a, b = geometry.edges.T
    delta = positions[b] - positions[a]
    lengths = np.linalg.norm(delta, axis=1)
    strain = lengths / geometry.lengths - 1
    if velocities is None:
        return strain, None
    strain_rate = np.sum(delta * (velocities[b] - velocities[a]), axis=1) / (
        np.maximum(lengths, 1e-12) * geometry.lengths
    )
    return strain, strain_rate


def _rest_geometry(env, count):
    # Optional metadata, no new backend methods or hidden-particle selection.
    rest = getattr(env, "rest_positions", None)
    edges = getattr(env, "edges", None)
    faces = getattr(env, "faces", None)
    if rest is None or (edges is None and faces is None):
        metadata = env.snapshot()
        rest = metadata.get("rest_positions", rest)
        edges = metadata.get("edges", edges)
        faces = metadata.get("faces", faces)
    if rest is None or (edges is None and faces is None):
        return None
    rest = np.asarray(rest, dtype=float)
    if edges is None:
        faces = np.asarray(faces, dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise PrimitiveFailure("invalid_rest_faces")
        edges = np.vstack((faces[:, :2], faces[:, 1:], faces[:, [0, 2]]))
    edges = np.asarray(edges, dtype=int)
    if (
        rest.shape != (count, 3)
        or not np.isfinite(rest).all()
        or edges.ndim != 2
        or edges.shape[1] != 2
        or np.any(edges < 0)
        or np.any(edges >= count)
    ):
        raise PrimitiveFailure("invalid_rest_geometry")
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    if not len(lengths) or np.any(lengths <= 0):
        raise PrimitiveFailure("invalid_rest_edge_lengths")
    return _RestGeometry(rest.copy(), edges, lengths)


def _rest_path(geometry, ids, count):
    if geometry is None:
        return None, None
    edges, lengths = geometry.edges, geometry.lengths
    graph: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for (a, b), length in zip(edges, lengths, strict=True):
        graph[a].append((int(b), float(length)))
        graph[b].append((int(a), float(length)))
    distances = np.full(count, np.inf)
    previous = np.full(count, -1, dtype=int)
    distances[ids[0]] = 0
    queue = [(0.0, ids[0])]
    while queue:
        d, a = heapq.heappop(queue)
        if d != distances[a]:
            continue
        if a == ids[1]:
            path = [a]
            while path[-1] != ids[0]:
                path.append(int(previous[path[-1]]))
            return np.array(path[::-1]), float(d)
        for b, length in graph[a]:
            if d + length < distances[b]:
                distances[b], previous[b] = d + length, a
                heapq.heappush(queue, (d + length, b))
    raise PrimitiveFailure("disconnected_rest_grasp_pair")


def _bezier(control, u):
    work = control.copy()
    for n in range(len(control) - 1, 0, -1):
        work[:n] = (1 - u) * work[:n] + u * work[1 : n + 1]
    return work[0]


def _trajectory(points, durations, speed, acceleration):
    """C2 quintic pieces; shared nonzero tangents through internal knots.

    Derivative control hulls give conservative *analytic* speed/acceleration bounds.
    Only the beginning and end of this entire trajectory intentionally stop.
    """
    points = np.asarray(points, dtype=float)
    durations = np.asarray(durations, dtype=float)
    tangents = np.zeros_like(points)
    for i in range(1, len(points) - 1):
        tangents[i] = (points[i + 1] - points[i - 1]) / (durations[i - 1] + durations[i])
        # A constant-height drag needs zero vertical tangent at the join, while
        # keeping its horizontal pass-through tangent. This prevents table undershoot.
        plateau = (points[i, :, 2] == points[i - 1, :, 2]) | (
            points[i + 1, :, 2] == points[i, :, 2]
        )
        tangents[i, plateau, 2] = 0
    controls = []
    for i, duration in enumerate(durations):
        a, b = points[i], points[i + 1]
        v, w = tangents[i], tangents[i + 1]
        controls.append(
            np.array(
                [
                    a,
                    a + v * duration / 5,
                    a + 2 * v * duration / 5,
                    b - 2 * w * duration / 5,
                    b - w * duration / 5,
                    b,
                ]
            )
        )
    vmax = max(
        np.linalg.norm(5 * np.diff(c, axis=0) / d, axis=-1).max()
        for c, d in zip(controls, durations, strict=True)
    )
    amax = max(
        np.linalg.norm(20 * np.diff(c, n=2, axis=0) / d**2, axis=-1).max()
        for c, d in zip(controls, durations, strict=True)
    )
    scale = max(1.0, vmax / speed, math.sqrt(amax / acceleration))
    return controls, durations * scale


class _Execution:
    def __init__(self, env, config, started, sim_started):
        self.env, self.c, self.started, self.sim_started = (
            env,
            config,
            started,
            sim_started,
        )
        self.dt = float(env.config.sim_dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise PrimitiveFailure("invalid_sim_dt")
        self.ids: tuple[int, ...] = ()
        self.grasp_ids: tuple[int, ...] = ()
        self.release_velocity_delta = None
        self.trace: list[dict[str, Any]] = []
        self.phases: list[str] = []
        self.geometry = None
        self.path = None
        self.rest_distance = None
        self.rest_flat_chord = None
        self.tauten_diagnostics: dict[str, Any] = {}
        self.target = None
        self.last_trace = -math.inf
        self.max_strain = None
        self.max_speed = 0.0
        self.command_speed = 0.0
        self.command_acceleration = 0.0
        self.previous_command_velocity = np.zeros((2, 3))
        self.settled = False
        self.release_coverage = None
        self.peak_settle_coverage = None

    def check(self, *, step=False):
        elapsed = float(self.env.simulated_time) - self.sim_started
        if not math.isfinite(elapsed) or elapsed < -1e-9:
            raise PrimitiveFailure("invalid_simulation_clock")
        if elapsed + (self.dt if step else 0) > self.c.max_simulated_seconds + 1e-9:
            raise DeadlineExceeded("simulated_deadline")
        if time.monotonic() - self.started >= self.c.max_wall_seconds:
            raise DeadlineExceeded("wall_deadline")

    def sample(self, phase, *, force=False):
        self.check()
        p, v = _state(self.env)
        if self.ids and tuple(self.env.attached_ids) != self.ids:
            raise PrimitiveFailure("attachment_lost_or_changed")
        speed = np.linalg.norm(v, axis=1)
        self.max_speed = max(self.max_speed, float(speed.max()))
        strain = None
        if self.geometry is not None:
            strain = float(np.max(_edge_strain(self.geometry, p)[0]))
            self.max_strain = max(self.max_strain or 0, strain)
            if strain > self.c.max_strain + 1e-6:
                raise PrimitiveFailure("cloth_strain_limit")
        now = float(self.env.simulated_time) - self.sim_started
        if force or now - self.last_trace >= self.c.trace_interval_seconds - 1e-9:
            coverage, area, _ = _coverage(self.env)
            if phase == "settle":
                self.peak_settle_coverage = max(self.peak_settle_coverage or coverage, coverage)
            near_table = p[:, 2] <= self.c.table_height_m + 0.01
            self.trace.append(
                {
                    "phase": phase,
                    "simulated_seconds": now,
                    "commanded_grasp_positions_m": None
                    if self.target is None
                    else self.target.tolist(),
                    "measured_grasp_positions_m": p[list(self.grasp_ids)].tolist()
                    if self.grasp_ids
                    else [],
                    "attachment_ids": list(self.env.attached_ids),
                    "cloth_positions_m": p.copy(),
                    "cloth_velocities_m_s": v.copy(),
                    "coverage": coverage,
                    "area_m2": area,
                    "max_edge_strain_proxy": strain,
                    "rms_speed_m_s": float(np.sqrt(np.mean(speed**2))),
                    "max_speed_m_s": float(speed.max()),
                    "table_proximity_fraction_proxy": float(np.mean(near_table)),
                }
            )
            self.last_trace = now
        return p, v

    def phase(self, name):
        self.phases.append(name)
        self.sample(name, force=True)

    def step(self, phase, target=None):
        self.check(step=True)
        if target is not None:
            target = np.asarray(target, dtype=float)
            if (
                not np.isfinite(target).all()
                or np.any(np.abs(target[:, :2]) > self.env.config.grasp_workspace_m / 2)
                or np.any(target[:, 2] < self.c.table_height_m)
            ):
                raise PrimitiveFailure("attachment_target_outside_workspace")
            velocity = (target - self.target) / self.dt
            acceleration = (velocity - self.previous_command_velocity) / self.dt
            speed = float(np.linalg.norm(velocity, axis=1).max())
            acceleration_norm = float(np.linalg.norm(acceleration, axis=1).max())
            if (
                speed > self.c.max_speed_m_s + 1e-6
                or acceleration_norm > self.c.max_acceleration_m_s2 + 1e-6
            ):
                raise PrimitiveFailure("command_kinematic_limit")
            self.env.set_attachment_targets(target)
            self.command_speed = max(self.command_speed, speed)
            self.command_acceleration = max(self.command_acceleration, acceleration_norm)
            self.target = target.copy()
            self.previous_command_velocity = velocity
        previous_time = float(self.env.simulated_time)
        self.env.step()
        if not math.isclose(
            float(self.env.simulated_time) - previous_time,
            self.dt,
            abs_tol=1e-8,
            rel_tol=1e-6,
        ):
            raise PrimitiveFailure("unexpected_backend_step_duration")
        return self.sample(phase)

    def follow(self, points, durations, phases):
        controls, durations = _trajectory(
            points, durations, self.c.max_speed_m_s, self.c.max_acceleration_m_s2 * 0.9
        )
        # Round the entire timeline once. Evaluate pieces by global time: never stop at waypoints.
        cumulative = np.cumsum(durations)
        total = float(cumulative[-1])
        count = math.ceil(total / self.dt)
        time_scale = count * self.dt / total
        phase_index = -1
        for step in range(1, count + 1):
            t = step * self.dt / time_scale
            index = min(int(np.searchsorted(cumulative, t, side="left")), len(controls) - 1)
            if index != phase_index:
                self.phase(phases[index])
                phase_index = index
            start = 0 if index == 0 else cumulative[index - 1]
            u = float(np.clip((t - start) / durations[index], 0, 1))
            self.step(phases[index], _bezier(controls[index], u))
        # One stationary sample completes endpoint deceleration, at an intended phase stop.
        self.step(phases[-1], np.asarray(points[-1]))

    def tauten(self):
        self.phase("tauten")
        p, _ = _state(self.env)
        a, b = p[list(self.ids)]
        axis = b - a
        axis[2] = 0
        span = float(np.linalg.norm(axis))
        if span <= 1e-9:
            raise PrimitiveFailure("degenerate_grasp_axis")
        axis /= span
        center = self.target.mean(axis=0)
        cap = float(self.env.config.max_grasp_separation_m)
        if self.rest_flat_chord is not None:
            # Flat rectangular material distance is the Euclidean rest chord. The
            # edge-graph path can have a large triangulation/Manhattan bias.
            cap = min(cap, self.rest_flat_chord * (1 + self.c.max_strain))
            goal = min(cap, self.rest_flat_chord * self.c.taut_rest_ratio)
        else:
            goal = min(cap, span + self.c.fallback_spread_m)
        if span > cap + 1e-6:
            raise PrimitiveFailure("initial_span_exceeds_rest_bound")
        goal = max(span, goal)
        # Feedback gates the smooth span-rate envelope using measured path sag and chord.
        duration = max(
            self.c.tauten_seconds,
            1.875 * (goal - span) / self.c.tauten_speed_m_s,
            math.sqrt(6 * (goal - span) / self.c.max_acceleration_m_s2),
        )
        count = max(2, math.ceil(duration / self.dt))
        duration = count * self.dt
        initial_span, rate = span, 0.0
        acceleration = self.c.max_acceleration_m_s2  # span acceleration; endpoints half
        soft_limit = 0.8 * self.c.max_strain
        braking = False
        self.tauten_diagnostics = {
            "material_bound": "rest_flat_chord" if self.rest_flat_chord is not None else "fallback",
            "initial_span_m": span,
            "target_span_m": goal,
            "span_cap_m": cap,
            "soft_strain_limit": soft_limit if self.geometry is not None else None,
            "stop_reason": "envelope_completed",
            "strain_brake": None,
        }
        # Match the tiny finite-difference residual from the completed lift.
        self.previous_command_velocity = np.zeros((2, 3))
        for step in range(1, count + 1):
            p, v = _state(self.env)
            measured_span = float(np.linalg.norm(p[self.ids[1]] - p[self.ids[0]]))
            slack = max(0.0, goal - measured_span)
            gate = min(1.0, slack / max(goal - initial_span, 1e-9))
            if self.path is not None:
                path_points = p[self.path]
                chord = p[self.ids[1]] - p[self.ids[0]]
                unit = chord / max(np.linalg.norm(chord), 1e-9)
                relative = path_points - p[self.ids[0]]
                sag = float(
                    np.linalg.norm(relative - np.outer(relative @ unit, unit), axis=1).max()
                )
                gate = min(gate, sag / self.c.taut_sag_tolerance_m)
            if self.geometry is not None and not braking:
                strain, strain_rate = _edge_strain(self.geometry, p, v)
                # Predict each edge over endpoint braking plus two physics ticks;
                # take the worst edge, never an average that hides local extension.
                horizon = rate / acceleration + 2 * self.dt
                predicted = float(np.max(strain + np.maximum(strain_rate, 0) * horizon))
                if predicted >= soft_limit:
                    braking = True
                    self.tauten_diagnostics["stop_reason"] = "local_strain_guard"
                    self.tauten_diagnostics["strain_brake"] = {
                        "simulated_seconds": float(self.env.simulated_time) - self.sim_started,
                        "max_edge_strain_proxy": float(np.max(strain)),
                        "predicted_edge_strain_proxy": predicted,
                        "span_rate_m_s": rate,
                        "prediction_horizon_seconds": horizon,
                    }
            u = step / count
            envelope = 30 * u**2 * (1 - u) ** 2 * (goal - initial_span) / duration
            remaining = max(0.0, goal - span)
            # Reserve a full tick and continuous stopping distance before the goal:
            # next_rate*dt + next_rate**2/(2*a) <= remaining.
            stopping_rate = math.sqrt((acceleration * self.dt) ** 2 + 2 * acceleration * remaining)
            stopping_rate = max(0.0, stopping_rate - acceleration * self.dt)
            desired_rate = 0.0 if braking else min(envelope * gate, stopping_rate)
            delta = acceleration * self.dt
            rate += float(np.clip(desired_rate - rate, -delta, delta))
            span += rate * self.dt
            if span > cap + 1e-9:
                raise PrimitiveFailure("tauten_rest_span_limit")
            self.step("tauten", np.array([center - axis * span / 2, center + axis * span / 2]))
            if braking and rate == 0.0:
                break
        self.step("tauten", self.target)
        self.tauten_diagnostics["final_span_m"] = span
        self.sample("tauten", force=True)

    def settle(self):
        self.phase("settle")
        start = float(self.env.simulated_time)
        stable = 0.0
        while (
            float(self.env.simulated_time) - start + self.dt <= self.c.settle_timeout_seconds + 1e-9
        ):
            _, v = self.step("settle")
            rms = float(np.sqrt(np.mean(np.sum(v**2, axis=1))))
            stable = stable + self.dt if rms < self.c.settle_rms_m_s else 0.0
            if stable + 1e-9 >= self.c.settle_hold_seconds:
                self.settled = True
                self.sample("settle", force=True)
                return
        raise PrimitiveFailure("settle_timeout")


def execute_action(
    env: ClothBackend, action: GraspAction, config: PrimitiveConfig | None = None
) -> ActionOutcome:
    """Execute without reset. Failures return invalid outcomes with terminal observation/evidence.

    An initial observation/coverage failure raises (no honest before-observation can be supplied).
    Cleanup still releases attachment. TimeoutError propagates to collector/supervisor owners.
    Missing terminal observations return after=None on an invalid outcome.
    Deadlines are cooperative around each backend operation.
    """
    c = config or PrimitiveConfig()
    started, sim_started = time.monotonic(), float(env.simulated_time)
    before = None
    coverage_before = None
    run = None
    errors = []
    reason, valid, truncated = "completed", False, False
    final_observation_valid = True
    released = False
    try:
        before = env.observe()
        coverage_before = _coverage(env)
        run = _Execution(env, c, started, sim_started)
        run.check()
        if action.primitive not in {"laydown", "fling"}:
            raise PrimitiveFailure("unknown_primitive")
        if tuple(env.attached_ids):
            raise PrimitiveFailure("backend_already_attached")
        p, _ = run.sample("before", force=True)
        uv = np.asarray(action.pixels, dtype=float)
        if uv.shape != (2, 2):
            raise PrimitiveFailure("invalid_action_pixels")
        xyz = pixels_to_world(before, uv)
        pixels = np.rint(uv).astype(int)
        mask = foreground_mask(before)
        if not mask[pixels[:, 1], pixels[:, 0]].all():
            raise PrimitiveFailure("grasp_not_rgb_foreground")
        separation = float(np.linalg.norm(xyz[1] - xyz[0]))
        if (
            np.linalg.norm(uv[1] - uv[0]) < 2
            or not env.config.min_grasp_separation_m
            <= separation
            <= env.config.max_grasp_separation_m
            or np.any(np.abs(xyz[:, :2]) > env.config.grasp_workspace_m / 2)
        ):
            raise PrimitiveFailure("infeasible_action_geometry")
        run.geometry = _rest_geometry(env, len(p))
        ids = tuple(env.attach(action.pixels))
        if (
            len(ids) != 2
            or len(set(ids)) != 2
            or any(i < 0 or i >= len(p) for i in ids)
            or tuple(env.attached_ids) != ids
        ):
            raise PrimitiveFailure("invalid_attachment_ids")
        run.ids = ids
        run.grasp_ids = ids
        run.path, run.rest_distance = _rest_path(run.geometry, ids, len(p))
        if run.geometry is not None:
            run.rest_flat_chord = float(
                np.linalg.norm(run.geometry.positions[ids[1]] - run.geometry.positions[ids[0]])
            )
            if run.rest_flat_chord <= 1e-12:
                raise PrimitiveFailure("degenerate_rest_grasp_chord")
        p, _ = _state(env)
        run.target = p[list(ids)].copy()
        run.phase("attach")
        lifted = run.target.copy()
        lifted[:, 2] = c.table_height_m + c.lift_height_m
        if np.any(lifted[:, 2] < run.target[:, 2]):
            raise PrimitiveFailure("grasp_above_lift_height")
        run.follow([run.target.copy(), lifted], [c.lift_seconds], ["lift"])
        run.tauten()
        start = run.target.copy()
        axis = start[1] - start[0]
        direction = np.array([-axis[1], axis[0], 0.0])
        direction /= max(np.linalg.norm(direction), 1e-9)
        if np.dot(start.mean(axis=0), direction) > 0:
            direction = -direction
        if action.primitive == "laydown":
            laid = start + direction * c.laydown_distance_m
            laid[:, 2] = c.table_height_m + c.release_height_m
            dragged = laid + direction * c.drag_distance_m
            run.follow(
                [start, laid, dragged],
                [c.laydown_seconds, c.drag_seconds],
                ["diagonal_laydown", "drag"],
            )
        else:
            swung = start + direction * c.swing_distance_m
            pulled = start - direction * c.pullback_distance_m
            pulled[:, 2] -= min(0.04, c.lift_height_m / 4)
            lower = start - direction * (c.pullback_distance_m / 2)
            lower[:, 2] = c.table_height_m + max(c.release_height_m, c.lift_height_m / 4)
            placed = start + direction * c.place_distance_m
            placed[:, 2] = c.table_height_m + c.release_height_m
            run.follow(
                [start, swung, pulled, lower, placed],
                [c.fling_segment_seconds] * 4,
                ["swing", "pullback", "lower", "place"],
            )
        run.phase("release")
        measured, release_velocity = _state(env)
        if np.any(measured[list(ids), 2] > c.table_height_m + 0.02):
            raise PrimitiveFailure("release_not_near_table")
        run.release_coverage = _coverage(env)[0]
        env.release()  # Backend must preserve velocities. Never write cloth positions/velocities.
        released = True
        run.ids = ()
        _, after_release_velocity = _state(env)
        run.release_velocity_delta = float(
            np.linalg.norm(after_release_velocity - release_velocity, axis=1).max()
        )
        if run.release_velocity_delta > 1e-6:
            raise PrimitiveFailure("release_changed_velocity")
        if tuple(env.attached_ids):
            raise PrimitiveFailure("release_left_attachments")
        run.sample("released", force=True)
        run.settle()
        valid = True
    except TimeoutError:
        # Collector/supervisor cancellation must reach its owner after finally cleanup.
        raise
    except Exception as exc:
        reason = (
            str(exc)
            if isinstance(exc, PrimitiveFailure)
            else f"unexpected_{type(exc).__name__}: {exc}"
        )
        errors.append({"type": type(exc).__name__, "message": str(exc)})
        truncated = isinstance(exc, DeadlineExceeded)
    finally:
        try:
            if not released or tuple(env.attached_ids):
                env.release()
            if tuple(env.attached_ids):
                raise PrimitiveFailure("cleanup_left_attachments")
        except TimeoutError:
            raise
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc), "stage": "release"})
            reason, valid = "release_cleanup_failed", False
    if before is None or coverage_before is None:
        raise PrimitiveFailure(f"initial_capture_failed: {reason}; errors={errors}")
    after = None
    try:
        after = env.observe()
        if after is None:
            raise PrimitiveFailure("terminal_observation_unavailable")
        p, v = _state(env)
        coverage_after = _coverage(env)
        if not math.isclose(coverage_before[2], coverage_after[2], rel_tol=1e-8, abs_tol=1e-12):
            raise PrimitiveFailure("flat_reference_changed")
    except TimeoutError:
        # Release cleanup has already completed before terminal capture begins.
        raise
    except Exception as exc:
        coverage_after = (math.nan, math.nan, coverage_before[2])
        final_observation_valid = False
        reason, valid = f"terminal_capture_failed_{type(exc).__name__}: {exc}", False
        errors.append({"type": type(exc).__name__, "message": str(exc), "stage": "terminal"})
        truncated |= isinstance(exc, DeadlineExceeded)
    if run:
        try:
            run.check()
        except PrimitiveFailure as exc:
            if valid:
                reason = str(exc)
            valid = False
            errors.append({"type": type(exc).__name__, "message": str(exc), "stage": "deadline"})
            truncated |= isinstance(exc, DeadlineExceeded)
    final = coverage_after[0]
    diagnostics = {
        "coverage_gain": final - coverage_before[0],
        "settled": bool(run and run.settled),
        "errors": errors,
        "terminal_observation_valid": final_observation_valid,
        "attachment_assisted": True,
        "contact_only": False,
        "behavior_passed": None,
        "depth_geometry_privileged": True,
        "rest_geodesic_distance_m": run.rest_distance if run else None,
        "graph_geodesic_distance": run.rest_distance if run else None,
        "rest_flat_chord_m": run.rest_flat_chord if run else None,
        "tauten": run.tauten_diagnostics if run else {},
        "strain_bound_available": bool(run and run.geometry is not None),
        "max_edge_strain_proxy": run.max_strain if run else None,
        "max_cloth_speed_m_s": run.max_speed if run else None,
        "max_command_speed_m_s": run.command_speed if run else None,
        "max_command_acceleration_m_s2": run.command_acceleration if run else None,
        "coverage_at_release": run.release_coverage if run else None,
        "release_velocity_delta_max_m_s": run.release_velocity_delta if run else None,
        "coverage_recoil_after_release": (
            run.release_coverage - final if run and run.release_coverage is not None else None
        ),
        "coverage_recoil_from_settle_peak": (
            max(run.peak_settle_coverage or final, final) - final if run else None
        ),
        "proxy_labels": {
            "max_edge_strain_proxy": "mesh edge extension/rest length minus one; not stress",
            "table_proximity_fraction_proxy": "z <= table+0.01m; not native contact",
            "tauten": "flat-rest chord target; graph path sag proxy; no force sensor",
            "strain_prediction": "edge extension/rate extrapolation over braking horizon; not stress",
            "missing_rest": "endpoint span fallback; no material strain guarantee",
        },
    }
    return ActionOutcome(
        before=before,
        after=after,
        action=action,
        coverage_before=coverage_before[0],
        coverage_after=final,
        area_before_m2=coverage_before[1],
        area_after_m2=coverage_after[1],
        flat_area_m2=coverage_before[2],
        execution_valid=valid,
        reason=reason,
        simulated_seconds=float(env.simulated_time) - sim_started,
        wall_seconds=time.monotonic() - started,
        terminated=not valid,
        truncated=truncated,
        diagnostics=diagnostics,
        evidence={
            "phases": run.phases if run else [],
            "phase_traces": run.trace if run else [],
            "terminal_state": {"positions_m": p.tolist(), "velocities_m_s": v.tolist()}
            if final_observation_valid
            else None,
            "config": asdict(c),
            "trace_interval_seconds": c.trace_interval_seconds,
            "source_adaptation": "FlingBot c5f35c785c6bc5d49282fb10117aa66ca04bf144",
        },
    )
