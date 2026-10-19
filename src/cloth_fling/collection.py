"""Bounded simulation-only repeated-action collection with durable execution intent."""

from __future__ import annotations

import copy
import json
import math
import platform
import signal
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .data import DatasetWriter
from .data.storage import jsonable, relative_path
from .provenance import config_provenance, execute_recorded, runtime_provenance


class CollectionTimeout(TimeoutError):
    pass


@contextmanager
def _deadline(deadline):
    """Interrupt Python simulator work; no background worker can keep acting on timeout."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CollectionTimeout("collection wall budget exhausted")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("collection requires an unused process real-time timer")

    def expire(signum, frame):
        raise CollectionTimeout("collection wall budget exhausted")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _finite_coverage(value):
    return float(value) if value is not None and math.isfinite(value) else None


def _initial_coverage(env, state, *, restored, metadata):
    """Read only. A restore needs fixture certification, never an implicit settle."""
    coverage = _finite_coverage(env.coverage()[0]) if hasattr(env, "coverage") else None
    velocities = np.asarray(state.get("velocities", []))
    finite_velocity = (
        velocities.ndim == 2
        and velocities.shape[1] == 3
        and len(velocities) > 0
        and np.isfinite(velocities).all()
    )
    rms = float(np.sqrt(np.mean(np.sum(velocities**2, axis=1)))) if finite_velocity else None
    released = not tuple(getattr(env, "attached_ids", ()))
    for key in ("attached_ids", "attached_particle_ids", "attachments"):
        if key in state:
            released = released and state[key] is not None and len(state[key]) == 0
    if "free" in state:
        released = released and bool(np.asarray(state["free"], dtype=bool).all())
    if "attachment_count" in state:
        released = released and state["attachment_count"] == 0
    certified = not restored or metadata.get("initial_settled") is True
    confirmed = certified and released and rms is not None and rms < 0.01 and coverage is not None
    source = "certified_restored_fixture" if restored else "successful_reset_contract"
    return coverage, rms, source if confirmed else None


def collect_episodes(
    env,
    output_dir,
    *,
    episodes=1,
    max_actions=10,
    seed=0,
    policy="uniform_valid",
    primitive="laydown",
    primitive_config=None,
    selector=None,
    executor=None,
    max_wall_seconds=600,
    reset_state=None,
    reset_kind="drop",
    record_video=False,
    appearance=None,
    recorder_factory=None,
    split=None,
    reset_group_id=None,
    cloth_id=None,
    episode_metadata=None,
    settled_coverage_threshold=None,
) -> dict:
    """Collect evolving-cloth episodes; see docs/dataset.md for interruption semantics.

    Injected selector: (observation, scene_config, *, policy, primitive, rng) -> action | None.
    Injected executor: (env, action, *, config) -> ActionOutcome. No uncertain call is retried.
    The wall budget covers all episodes and bounds each environment/selection/execution call.
    Durable writes finish outside the timer. Python cannot preempt a native GPU kernel.
    max_wall_seconds is the whole-call budget, independent of the primitive's action budget.
    Restored initial settlement requires episode_metadata['initial_settled'] is True and
    a read-only RMS velocity check. No settling steps are added by the collector.
    """
    if (
        isinstance(episodes, bool)
        or not isinstance(episodes, int)
        or episodes < 0
        or isinstance(max_actions, bool)
        or not isinstance(max_actions, int)
        or not 0 <= max_actions <= 10
    ):
        raise ValueError("episodes must be nonnegative; max_actions must be an integer in [0, 10]")
    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive and finite")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (
        ("split", split),
        ("reset_group_id", reset_group_id),
        ("cloth_id", cloth_id),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string or None")
    if episode_metadata is not None and not isinstance(episode_metadata, Mapping):
        raise TypeError("episode_metadata must be a mapping or None")
    metadata = jsonable(copy.deepcopy(dict(episode_metadata or {})))
    if settled_coverage_threshold is not None and (
        isinstance(settled_coverage_threshold, bool)
        or not isinstance(settled_coverage_threshold, (int, float))
        or not math.isfinite(settled_coverage_threshold)
        or not 0 < settled_coverage_threshold <= 1
    ):
        raise ValueError("settled_coverage_threshold must be in (0, 1] or None")
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        raise RuntimeError("bounded collection requires a POSIX main thread")
    if selector is None:
        from .selection import select_action

        selector = select_action
    if executor is None:
        from .sim.primitives import execute_action

        executor = execute_action
    started = time.monotonic()
    reset_source = copy.deepcopy(reset_state)
    deadline = started + max_wall_seconds
    report = {
        "simulation_only": True,
        "recordings": [],
        "runtime_provenance": runtime_provenance(),
        "provenance": config_provenance(
            env,
            appearance=appearance,
            primitive_config=primitive_config,
            reset_source=reset_source,
        ),
        "attempts": 0,
        "valid": 0,
        "invalid": 0,
        "reset_failures": 0,
        "episodes": 0,
        "episode_reports": [],
        "stop_reason": "completed",
        "episode_metadata": metadata,
        "settled_coverage_threshold": settled_coverage_threshold,
        "timings": {
            "reset_seconds": 0.0,
            "selection_seconds": 0.0,
            "execution_seconds": 0.0,
            "wall_seconds": 0.0,
            "storage_seconds": 0.0,
            "by_recording": {
                mode: {
                    "attempts": 0,
                    "selection_seconds": 0.0,
                    "execution_seconds": 0.0,
                    "simulated_seconds": 0.0,
                }
                for mode in ("recorded", "unrecorded")
            },
        },
    }
    mode_timings = report["timings"]["by_recording"]["recorded" if record_video else "unrecorded"]

    def persist(method, *args, **kwargs):
        tick = time.monotonic()
        try:
            return method(*args, **kwargs)
        finally:
            report["timings"]["storage_seconds"] += time.monotonic() - tick

    def finish_report(writer, episode_report):
        episode = writer.manifest["episodes"][episode_report["episode_id"]]
        for key in (
            "split",
            "reset_group_id",
            "cloth_id",
            "reset_id",
            "config_sha256",
            "end_reason",
        ):
            episode_report[key] = episode[key]
        episode_report["lifecycle"] = (
            "initially_unfolded"
            if episode["end_reason"] == "initially_unfolded"
            else "unfolded_after_action"
            if episode["end_reason"] == "settled_coverage_threshold"
            else "stopped_without_threshold"
        )
        persist(writer.write_collection_summary, episode_report["episode_id"], episode_report)

    def snapshot():
        try:
            with _deadline(deadline):
                return env.snapshot()
        except CollectionTimeout:
            return None
        except Exception as exc:  # noqa: BLE001 - persist backend support failures.
            return {"snapshot_error": f"{type(exc).__name__}: {exc}"}

    def cleanup():
        # Assisted-release cleanup only: no step, restore, reset, or execution retry.
        evidence = {"release_attempted": True, "release_error": None}
        after = state = None
        cleanup_deadline = time.monotonic() + 1.0
        try:
            with _deadline(cleanup_deadline):
                env.release()
        except Exception as exc:  # noqa: BLE001 - cleanup errors are evidence.
            evidence["release_error"] = f"{type(exc).__name__}: {exc}"
        for name in ("observe", "snapshot"):
            try:
                with _deadline(cleanup_deadline):
                    value = getattr(env, name)()
                if name == "observe":
                    after = value
                else:
                    state = value
            except Exception as exc:  # noqa: BLE001 - capture read-only cleanup failures.
                evidence[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        return after, state, evidence

    versions = {"python": platform.python_version()}
    for package in (
        "cloth-fling",
        "numpy",
        "pillow",
        "pyarrow",
        "genesis-world",
        "torch",
        "quadrants",
        "trimesh",
        "pymeshlab",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    with DatasetWriter(
        output_dir,
        metadata={
            "collector": "collect_episodes_v1",
            "seed": seed,
            "policy": policy,
            "primitive": primitive,
            "primitive_config": primitive_config,
            "max_actions": max_actions,
            "max_wall_seconds": max_wall_seconds,
            "reset_kind": reset_kind,
            "uses_saved_reset": reset_source is not None,
            "runtime_package_versions": versions,
            "runtime_provenance": report["runtime_provenance"],
            "configuration_provenance": report["provenance"],
            "record_video": record_video,
            "split": split or "train",
            "episode_metadata": metadata,
            "settled_coverage_threshold": settled_coverage_threshold,
        },
    ) as writer:
        report["run_id"] = writer.run_id
        existing_files = (
            set(writer.manifest["files"]) if len(writer.manifest["runs"]) > 1 else set()
        )
        for _ in range(episodes):
            if time.monotonic() >= deadline:
                report["stop_reason"] = "max_wall_seconds"
                break
            # Continuing output gets a fresh episode ID/seed and never replays old intent.
            episode_seed = seed + len(writer.manifest["episodes"])
            rng = np.random.default_rng(episode_seed)
            config = env.config.to_dict() if hasattr(env.config, "to_dict") else env.config
            episode_report = {
                "episode_id": None,
                "seed": episode_seed,
                "attempts": 0,
                "valid": 0,
                "invalid": 0,
                "end_reason": None,
                "metadata": copy.deepcopy(metadata),
                "record_video": bool(record_video),
                "settled_coverage_threshold": settled_coverage_threshold,
                "initial_coverage": None,
                "initial_confirmed_settled_coverage": None,
                "initial_settlement_source": None,
                "initial_rms_speed_m_s": None,
                "initial_observation_id": None,
                "last_confirmed_settled_coverage": None,
                "last_confirmed_settled_observation_id": None,
                "last_confirmed_settlement_source": None,
                "last_observed_coverage": None,
                "last_observed_execution_valid": None,
                "last_observation_id": None,
            }
            tick = time.monotonic()
            try:
                with _deadline(deadline):
                    current = (
                        env.restore(copy.deepcopy(reset_source))
                        if reset_source is not None
                        else env.reset(seed=episode_seed, kind=reset_kind)
                    )
                with _deadline(deadline):
                    observed_reset_state = env.snapshot()
                    initial_coverage, initial_rms, initial_source = _initial_coverage(
                        env,
                        observed_reset_state,
                        restored=reset_source is not None,
                        metadata=metadata,
                    )
            except Exception as exc:  # noqa: BLE001 - persist reset preparation failures.
                _, state, evidence = cleanup()
                reason = "reset_timeout" if isinstance(exc, CollectionTimeout) else "reset_failed"
                evidence["reset_error"] = f"{type(exc).__name__}: {exc}"
                episode_id = persist(
                    writer.write_reset_failure,
                    seed=episode_seed,
                    config=config,
                    reason=reason,
                    state=state,
                    reset_source=reset_source,
                    reset_kind=reset_kind,
                    evidence=evidence,
                    split=split,
                    reset_group_id=reset_group_id,
                    cloth_id=cloth_id,
                    episode_metadata=metadata,
                )
                report["reset_failures"] += 1
                report["episodes"] += 1
                episode_report["episode_id"] = episode_id
                report["episode_reports"].append(episode_report)
                finish_report(writer, episode_report)
                report["stop_reason"] = reason
                break
            finally:
                report["timings"]["reset_seconds"] += time.monotonic() - tick
            episode_id = persist(
                writer.write_episode_reset,
                current,
                seed=episode_seed,
                config=config,
                state=observed_reset_state,
                reset_source=reset_source,
                reset_kind=reset_kind,
                split=split,
                reset_group_id=reset_group_id,
                cloth_id=cloth_id,
                episode_metadata=metadata,
            )
            report["episodes"] += 1
            episode_report.update(
                episode_id=episode_id,
                initial_coverage=initial_coverage,
                initial_confirmed_settled_coverage=initial_coverage if initial_source else None,
                initial_settlement_source=initial_source,
                initial_rms_speed_m_s=initial_rms,
                initial_observation_id=current.observation_id,
                last_confirmed_settled_coverage=initial_coverage if initial_source else None,
                last_confirmed_settled_observation_id=current.observation_id
                if initial_source
                else None,
                last_confirmed_settlement_source=initial_source,
                last_observed_coverage=initial_coverage,
                last_observation_id=current.observation_id,
            )
            report["episode_reports"].append(episode_report)
            initially_unfolded = (
                settled_coverage_threshold is not None
                and initial_source is not None
                and initial_coverage >= settled_coverage_threshold
            )
            for step in range(0 if initially_unfolded else max_actions):
                if time.monotonic() >= deadline:
                    persist(
                        writer.finish_episode, episode_id, reason="max_wall_seconds", truncated=True
                    )
                    break
                tick = time.monotonic()
                try:
                    with _deadline(deadline):
                        action = selector(
                            current, env.config, policy=policy, primitive=primitive, rng=rng
                        )
                except CollectionTimeout:
                    persist(
                        writer.finish_episode,
                        episode_id,
                        reason="selection_timeout",
                        truncated=True,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - selection failure closes the episode.
                    persist(
                        writer.finish_episode,
                        episode_id,
                        reason="no_valid_pair"
                        if isinstance(exc, ValueError)
                        else "selection_error",
                        terminated=True,
                        final_state=snapshot(),
                        evidence={"selection_error": f"{type(exc).__name__}: {exc}"},
                    )
                    break
                finally:
                    elapsed = time.monotonic() - tick
                    report["timings"]["selection_seconds"] += elapsed
                    mode_timings["selection_seconds"] += elapsed
                if action is None:
                    persist(
                        writer.finish_episode,
                        episode_id,
                        reason="no_valid_pair",
                        terminated=True,
                        final_state=snapshot(),
                    )
                    break
                if time.monotonic() >= deadline:
                    persist(
                        writer.finish_episode, episode_id, reason="max_wall_seconds", truncated=True
                    )
                    break
                persist(writer.begin_attempt, episode_id, action, before=current)
                tick = time.monotonic()
                execution_elapsed = None
                confirmed_settled = False
                try:
                    with _deadline(deadline):
                        if record_video:
                            outcome = execute_recorded(
                                env,
                                action,
                                config=primitive_config,
                                executor=executor,
                                output_dir=output_dir,
                                action_id=f"episode-{episode_id}-action-{step}",
                                appearance=appearance,
                                recorder_factory=recorder_factory,
                            )
                        else:
                            outcome = executor(env, action, config=primitive_config)
                    execution_elapsed = time.monotonic() - tick
                    # The authoritative pre-observation is the exact prior after object.
                    outcome = replace(outcome, before=current)
                except CollectionTimeout as exc:
                    execution_elapsed = time.monotonic() - tick
                    after, state, evidence = cleanup()
                    if hasattr(exc, "recording_evidence"):
                        evidence["recording"] = exc.recording_evidence
                        report["recordings"].append(exc.recording_evidence)
                    row = persist(
                        writer.append_outcome,
                        episode_id,
                        reason="execution_timeout",
                        terminated=False,
                        truncated=True,
                        end_reason="max_wall_seconds",
                        after=after,
                        final_state=state,
                        failure_evidence=evidence,
                    )
                except Exception as exc:  # noqa: BLE001 - every attempted action needs a row.
                    execution_elapsed = time.monotonic() - tick
                    after, state, evidence = cleanup()
                    if hasattr(exc, "recording_evidence"):
                        evidence["recording"] = exc.recording_evidence
                        report["recordings"].append(exc.recording_evidence)
                    row = persist(
                        writer.append_outcome,
                        episode_id,
                        after=after,
                        final_state=state,
                        failure_evidence=evidence,
                        reason=f"execution_exception: {type(exc).__name__}: {exc}",
                        terminated=True,
                        end_reason="execution_invalid",
                    )
                else:
                    if record_video:
                        report["recordings"].append(outcome.evidence["recording"])
                    cleanup_evidence = None
                    if not outcome.execution_valid:
                        after, final_state, cleanup_evidence = cleanup()
                        if (
                            after is not None
                            or outcome.diagnostics.get("terminal_observation_valid") is False
                        ):
                            outcome = replace(outcome, after=after)
                    else:
                        final_state = snapshot()
                    timed_out = time.monotonic() >= deadline
                    limited = step + 1 == max_actions
                    env_ended = (
                        outcome.terminated or outcome.truncated or not outcome.execution_valid
                    )
                    confirmed_settled = (
                        outcome.execution_valid
                        and outcome.diagnostics.get("settled") is True
                        and outcome.diagnostics.get("terminal_observation_valid") is not False
                        and outcome.after is not None
                        and outcome.after.observation_id != current.observation_id
                        and _finite_coverage(outcome.coverage_after) is not None
                    )
                    threshold_reached = (
                        confirmed_settled
                        and settled_coverage_threshold is not None
                        and outcome.coverage_after >= settled_coverage_threshold
                        and not timed_out
                        and not env_ended
                    )
                    row = persist(
                        writer.append_outcome,
                        episode_id,
                        outcome,
                        final_state=final_state,
                        failure_evidence=cleanup_evidence,
                        terminated=outcome.terminated or threshold_reached,
                        truncated=outcome.truncated
                        or ((timed_out or limited) and not env_ended and not threshold_reached),
                        end_reason=(
                            "execution_invalid"
                            if not outcome.execution_valid
                            else "environment_terminated"
                            if env_ended
                            else "max_wall_seconds"
                            if timed_out
                            else "settled_coverage_threshold"
                            if threshold_reached
                            else "max_actions"
                            if limited
                            else None
                        ),
                    )
                    current = outcome.after
                finally:
                    elapsed = (
                        execution_elapsed
                        if execution_elapsed is not None
                        else time.monotonic() - tick
                    )
                    report["timings"]["execution_seconds"] += elapsed
                    mode_timings["execution_seconds"] += elapsed
                report["attempts"] += 1
                mode_timings["attempts"] += 1
                if row["simulated_seconds"] is not None:
                    mode_timings["simulated_seconds"] += row["simulated_seconds"]
                episode_report["attempts"] += 1
                key = "valid" if row["execution_valid"] else "invalid"
                report[key] += 1
                episode_report[key] += 1
                episode_report.update(
                    last_observed_coverage=row["coverage_after"],
                    last_observed_execution_valid=row["execution_valid"],
                    last_observation_id=row["after_observation_id"],
                )
                if row["execution_valid"] and confirmed_settled:
                    episode_report.update(
                        last_confirmed_settled_coverage=row["coverage_after"],
                        last_confirmed_settled_observation_id=row["after_observation_id"],
                        last_confirmed_settlement_source="valid_settled_action",
                    )
                if row["terminated"] or row["truncated"]:
                    break
            else:
                # Includes the explicitly allowed zero-action episode.
                persist(
                    writer.finish_episode,
                    episode_id,
                    reason="initially_unfolded" if initially_unfolded else "max_actions",
                    terminated=initially_unfolded,
                    truncated=not initially_unfolded,
                    final_state=snapshot(),
                )
            finish_report(writer, episode_report)
            if time.monotonic() >= deadline:
                report["stop_reason"] = "max_wall_seconds"
                break
        report["storage"] = {
            "registered_file_count": len(writer.manifest["files"]),
            "registered_bytes_total": sum(
                info["size_bytes"] for info in writer.manifest["files"].values()
            ),
            "registered_bytes_added": sum(
                info["size_bytes"]
                for path, info in writer.manifest["files"].items()
                if path not in existing_files
            ),
            "external_recording_bytes": 0,
            "external_recording_file_count": 0,
            "external_recording_stat_errors": [],
        }
        recording_files = {
            item["relative_file"]
            for recording in report["recordings"]
            for item in recording.get("files", [])
        }
        for path in recording_files:
            try:
                size = relative_path(Path(output_dir).resolve(), path).stat().st_size
                report["storage"]["external_recording_bytes"] += size
                report["storage"]["external_recording_file_count"] += 1
            except (OSError, ValueError) as exc:
                report["storage"]["external_recording_stat_errors"].append(f"{path}: {exc}")
    report["timings"]["wall_seconds"] = time.monotonic() - started
    destination = Path(output_dir) / f"collection-report-{report['run_id']}.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
