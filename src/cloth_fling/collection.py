"""Bounded simulation-only repeated-action collection with durable execution intent."""

from __future__ import annotations

import copy
import json
import math
import platform
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .data import DatasetWriter
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
) -> dict:
    """Collect evolving-cloth episodes; see docs/dataset.md for interruption semantics.

    Injected selector: (observation, scene_config, *, policy, primitive, rng) -> action | None.
    Injected executor: (env, action, *, config) -> ActionOutcome. No uncertain call is retried.
    The wall budget covers all episodes and bounds each environment/selection/execution call.
    Durable writes finish outside the timer. Python cannot preempt a native GPU kernel.
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
        "timings": {
            "reset_seconds": 0.0,
            "selection_seconds": 0.0,
            "execution_seconds": 0.0,
            "wall_seconds": 0.0,
        },
    }

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
        },
    ) as writer:
        report["run_id"] = writer.run_id
        for _ in range(episodes):
            if time.monotonic() >= deadline:
                report["stop_reason"] = "max_wall_seconds"
                break
            # Continuing output gets a fresh episode ID/seed and never replays old intent.
            episode_seed = seed + len(writer.manifest["episodes"])
            rng = np.random.default_rng(episode_seed)
            config = env.config.to_dict() if hasattr(env.config, "to_dict") else env.config
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
            except Exception as exc:  # noqa: BLE001 - persist reset preparation failures.
                _, state, evidence = cleanup()
                reason = "reset_timeout" if isinstance(exc, CollectionTimeout) else "reset_failed"
                evidence["reset_error"] = f"{type(exc).__name__}: {exc}"
                episode_id = writer.write_reset_failure(
                    seed=episode_seed,
                    config=config,
                    reason=reason,
                    state=state,
                    reset_source=reset_source,
                    reset_kind=reset_kind,
                    evidence=evidence,
                )
                report["reset_failures"] += 1
                report["episodes"] += 1
                report["episode_reports"].append(
                    {
                        "episode_id": episode_id,
                        "seed": episode_seed,
                        "attempts": 0,
                        "valid": 0,
                        "invalid": 0,
                        "end_reason": reason,
                    }
                )
                report["stop_reason"] = reason
                break
            finally:
                report["timings"]["reset_seconds"] += time.monotonic() - tick
            episode_id = writer.write_episode_reset(
                current,
                seed=episode_seed,
                config=config,
                state=observed_reset_state,
                reset_source=reset_source,
                reset_kind=reset_kind,
            )
            report["episodes"] += 1
            episode_report = {
                "episode_id": episode_id,
                "seed": episode_seed,
                "attempts": 0,
                "valid": 0,
                "invalid": 0,
                "end_reason": None,
            }
            report["episode_reports"].append(episode_report)
            for step in range(max_actions):
                if time.monotonic() >= deadline:
                    writer.finish_episode(episode_id, reason="max_wall_seconds", truncated=True)
                    break
                tick = time.monotonic()
                try:
                    with _deadline(deadline):
                        action = selector(
                            current, env.config, policy=policy, primitive=primitive, rng=rng
                        )
                except CollectionTimeout:
                    writer.finish_episode(episode_id, reason="selection_timeout", truncated=True)
                    break
                except Exception as exc:  # noqa: BLE001 - selection failure closes the episode.
                    writer.finish_episode(
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
                    report["timings"]["selection_seconds"] += time.monotonic() - tick
                if action is None:
                    writer.finish_episode(
                        episode_id, reason="no_valid_pair", terminated=True, final_state=snapshot()
                    )
                    break
                if time.monotonic() >= deadline:
                    writer.finish_episode(episode_id, reason="max_wall_seconds", truncated=True)
                    break
                writer.begin_attempt(episode_id, action, before=current)
                tick = time.monotonic()
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
                    # The authoritative pre-observation is the exact prior after object.
                    outcome = replace(outcome, before=current)
                except CollectionTimeout as exc:
                    after, state, evidence = cleanup()
                    if hasattr(exc, "recording_evidence"):
                        evidence["recording"] = exc.recording_evidence
                        report["recordings"].append(exc.recording_evidence)
                    row = writer.append_outcome(
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
                    after, state, evidence = cleanup()
                    if hasattr(exc, "recording_evidence"):
                        evidence["recording"] = exc.recording_evidence
                        report["recordings"].append(exc.recording_evidence)
                    row = writer.append_outcome(
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
                    row = writer.append_outcome(
                        episode_id,
                        outcome,
                        final_state=final_state,
                        failure_evidence=cleanup_evidence,
                        truncated=outcome.truncated or ((timed_out or limited) and not env_ended),
                        end_reason=(
                            "execution_invalid"
                            if not outcome.execution_valid
                            else "environment_terminated"
                            if env_ended
                            else "max_wall_seconds"
                            if timed_out
                            else "max_actions"
                            if limited
                            else None
                        ),
                    )
                    current = outcome.after
                finally:
                    report["timings"]["execution_seconds"] += time.monotonic() - tick
                report["attempts"] += 1
                episode_report["attempts"] += 1
                key = "valid" if row["execution_valid"] else "invalid"
                report[key] += 1
                episode_report[key] += 1
                if row["terminated"] or row["truncated"]:
                    break
            else:
                # Includes the explicitly allowed zero-action episode.
                writer.finish_episode(
                    episode_id, reason="max_actions", truncated=True, final_state=snapshot()
                )
            episode = writer.manifest["episodes"][episode_id]
            episode_report["end_reason"] = episode["end_reason"]
            if time.monotonic() >= deadline:
                report["stop_reason"] = "max_wall_seconds"
                break
    report["timings"]["wall_seconds"] = time.monotonic() - started
    destination = Path(output_dir) / f"collection-report-{report['run_id']}.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
