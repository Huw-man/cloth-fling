"""Frozen Milestone 3 reset evidence, durable budgets, and offline review.

No simulator is constructed here. ``prepare_benchmark`` operates only on the supplied
exclusive-lease environment; all other entrypoints are offline.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np

from .data.storage import (
    atomic_write,
    file_sha256,
    json_bytes,
    load_json,
    read_bundle,
    relative_path,
    sha256,
    write_bundle,
)
from .provenance import config_provenance

MANIFEST = "benchmark-manifest.json"
CLOTH_ID = "synthetic-square-0.50m-v1"
SELECTORS = ("uniform_valid", "geometry_guided")
LIMITS = {
    "prepare_fold": 3,
    "prepare_drop": 24,
    "diagnostic_laydown": 26,
    "diagnostic_fling": 6,
    "diagnostic_continuity": 8,
    "pilot": 120,
    "simulator": None,
}
WALL_LIMIT = 6 * 60 * 60


@contextmanager
def _locked(root, filename):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with relative_path(root, filename).open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"active benchmark operation: {filename}") from None
        yield root


def _digest(value):
    return sha256(json_bytes(value))


def _sealed_write(root, reference, value):
    """Exclusive immutable payload and digest seal; failed half-write is fail-closed."""
    if (
        relative_path(root, reference).exists()
        or relative_path(root, reference + ".sha256").exists()
    ):
        raise FileExistsError(reference)
    data = json_bytes(value)
    atomic_write(root, reference, data)
    atomic_write(root, reference + ".sha256", (sha256(data) + "\n").encode())


def _sealed_load(root, reference):
    if (
        file_sha256(relative_path(root, reference))
        != relative_path(root, reference + ".sha256").read_text().strip()
    ):
        raise ValueError(f"frozen document hash mismatch: {reference}")
    return load_json(root, reference)


def benchmark_identity(env, *, appearance=None):
    """Frozen scene, appearance, runtime/numerical and rendering identities."""
    result = config_provenance(env, appearance=appearance)
    for key in ("primitive_config", "source_reset_sha256", "source_snapshot_identity"):
        result.pop(key)
    physics = result["physics_config"]
    if physics.get("contact_profile") != "native_coulomb_v2" or physics["cloth_size_m"] != 0.5:
        raise ValueError("benchmark requires native-v2 and one synthetic 0.50m square")
    runtime = result["runtime_identity"]
    if not runtime or runtime.get("engine_version") != "1.4.4":
        raise ValueError("benchmark requires explicit Genesis 1.4.4 runtime identity")
    return result


class BudgetLedger:
    """Durable reservation before execution; uncertain calls consume their full allowance.

    Reserve the complete simulator operation (reset/restore/selection/action/capture)
    wall allowance, then enforce it with the existing deadline guard. An interruption
    leaves a pending reservation and cannot reset attempts or wall usage. ``abandon``
    closes it charging its complete allowance; it never authorizes replay.
    """

    def __init__(self, output_dir):
        self.root = Path(output_dir).resolve()

    def _read(self):
        path = self.root / "benchmark-budget.json"
        if not path.exists():
            return {"schema_version": 1, "entries": [], "pilot_episodes": []}
        result = load_json(self.root, path.name)
        if result.get("schema_version") != 1:
            raise ValueError("unsupported benchmark ledger")
        return result

    def _write(self, ledger):
        atomic_write(self.root, "benchmark-budget.json", json_bytes(ledger))

    @staticmethod
    def _status(ledger):
        entries = ledger["entries"]
        counts = {category: sum(e["category"] == category for e in entries) for category in LIMITS}
        wall = sum(e.get("wall_seconds", e["reserved_wall_seconds"]) for e in entries)
        if wall > WALL_LIMIT or any(
            limit is not None and counts[category] > limit for category, limit in LIMITS.items()
        ):
            raise ValueError("benchmark ledger exceeds immutable budgets")
        return {
            "counts": counts,
            "charged_wall_seconds": wall,
            "remaining_wall_seconds": WALL_LIMIT - wall,
            "pending": [e for e in entries if e["status"] == "reserved"],
            "pilot_episodes": copy.deepcopy(ledger["pilot_episodes"]),
        }

    def status(self):
        with _locked(self.root, ".benchmark-budget.lock"):
            return self._status(self._read())

    def reserve(self, category, *, wall_seconds, key=None, metadata=None):
        if category not in LIMITS:
            raise ValueError("unknown budget category")
        if isinstance(wall_seconds, bool) or not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("wall reservation must be finite and positive")
        metadata = copy.deepcopy(metadata or {})
        with _locked(self.root, ".benchmark-budget.lock"):
            ledger = self._read()
            status = self._status(ledger)
            if status["pending"]:
                raise RuntimeError("unresolved reservation; reconcile without replay")
            if key is not None and any(e["key"] == key for e in ledger["entries"]):
                raise ValueError("reservation key already consumed")
            limit = LIMITS[category]
            if limit is not None and status["counts"][category] >= limit:
                raise RuntimeError(f"{category} attempt budget exhausted")
            if wall_seconds > status["remaining_wall_seconds"]:
                raise RuntimeError("cumulative simulator wall budget exhausted")
            if category.startswith("diagnostic"):
                if metadata.get("split") == "heldout":
                    raise ValueError("heldout is forbidden for qualification/tuning")
                if any(e["split"] == "heldout" for e in ledger["pilot_episodes"]):
                    raise ValueError("diagnostic tuning is sealed after heldout starts")
                if metadata.get("fixture_id"):
                    manifest = load_benchmark(self.root)
                    fixture = _fixture(manifest, metadata["fixture_id"])
                    if fixture["split"] != "development":
                        raise ValueError("heldout fixture cannot be used for qualification")
            if category == "pilot":
                self._check_pilot_attempt(ledger, metadata)
            token = uuid4().hex
            ledger["entries"].append(
                {
                    "token": token,
                    "key": key,
                    "category": category,
                    "reserved_wall_seconds": float(wall_seconds),
                    "status": "reserved",
                    "reserved_unix": time.time(),
                    "metadata": metadata,
                }
            )
            self._write(ledger)
            return token

    def finish(self, token, *, wall_seconds, result=None):
        with _locked(self.root, ".benchmark-budget.lock"):
            ledger = self._read()
            entry = next(e for e in ledger["entries"] if e["token"] == token)
            if entry["status"] != "reserved":
                raise ValueError("reservation already finished")
            if (
                not math.isfinite(wall_seconds)
                or not 0 <= wall_seconds <= entry["reserved_wall_seconds"]
            ):
                raise ValueError("elapsed wall must fit reservation; retain pending on overrun")
            entry.update(
                status="finished",
                wall_seconds=float(wall_seconds),
                result=result,
                finished_unix=time.time(),
            )
            self._write(ledger)
            return copy.deepcopy(entry)

    def abandon(self, token, *, result=None):
        """Charge an uncertain/interrupted operation fully; no automatic resend."""
        with _locked(self.root, ".benchmark-budget.lock"):
            ledger = self._read()
            entry = next(e for e in ledger["entries"] if e["token"] == token)
            if entry["status"] != "reserved":
                raise ValueError("reservation already finished")
            entry.update(
                status="abandoned",
                wall_seconds=entry["reserved_wall_seconds"],
                result=result,
                finished_unix=time.time(),
            )
            self._write(ledger)
            return copy.deepcopy(entry)

    def _check_pilot_attempt(self, ledger, metadata):
        token = metadata.get("episode_token")
        episode = next((e for e in ledger["pilot_episodes"] if e["token"] == token), None)
        if episode is None or episode["status"] != "claimed":
            raise ValueError("pilot attempt requires an active claimed episode")
        for key in ("fixture_id", "selector", "split"):
            if key in metadata and metadata[key] != episode[key]:
                raise ValueError("pilot attempt/episode identity mismatch")
        prior = [
            e
            for e in ledger["entries"]
            if e["category"] == "pilot" and e["metadata"].get("episode_token") == token
        ]
        if len(prior) >= 5:
            raise RuntimeError("pilot episode action limit exhausted")

    def claim_pilot(self, fixture_id, selector, *, qualification_sha256):
        """Irreversibly claim one paired episode BEFORE restore, including zero-action failures."""
        manifest = load_benchmark(self.root)
        qualification = load_qualification(self.root)
        if not qualification["passed"] or qualification_sha256 != _digest(qualification):
            raise ValueError("pilot requires matching passed qualification")
        fixture = _fixture(manifest, fixture_id)
        if fixture["kind"] != "drop" or selector not in SELECTORS:
            raise ValueError("pilot requires a drop fixture and a declared selector")
        with _locked(self.root, ".benchmark-budget.lock"):
            ledger = self._read()
            if self._status(ledger)["pending"]:
                raise RuntimeError("unresolved simulator reservation")
            episodes = ledger["pilot_episodes"]
            if any(e["fixture_id"] == fixture_id and e["selector"] == selector for e in episodes):
                raise ValueError("pilot pair already consumed; no heldout reruns")
            if any(e["status"] == "claimed" for e in episodes):
                raise RuntimeError("finish the previously claimed pilot episode")
            if fixture["split"] == "heldout":
                expected = {
                    (f["fixture_id"], s)
                    for f in manifest["fixtures"]
                    if f["kind"] == "drop" and f["split"] == "development"
                    for s in SELECTORS
                }
                completed = {
                    (e["fixture_id"], e["selector"]) for e in episodes if e["status"] == "finished"
                }
                if not expected.issubset(completed):
                    raise ValueError("complete development pilot before heldout")
            elif any(e["split"] == "heldout" for e in episodes):
                raise ValueError("development execution/tuning forbidden after heldout starts")
            token = uuid4().hex
            episodes.append(
                {
                    "token": token,
                    "fixture_id": fixture_id,
                    "selector": selector,
                    "split": fixture["split"],
                    "state_content_sha256": fixture["state_content_sha256"],
                    "qualification_sha256": qualification_sha256,
                    "status": "claimed",
                }
            )
            self._write(ledger)
            return token

    def finish_pilot(self, token, *, result=None):
        with _locked(self.root, ".benchmark-budget.lock"):
            ledger = self._read()
            if self._status(ledger)["pending"]:
                raise RuntimeError("finish or abandon action reservation first")
            episode = next(e for e in ledger["pilot_episodes"] if e["token"] == token)
            if episode["status"] != "claimed":
                raise ValueError("pilot episode already finished")
            episode.update(status="finished", result=result)
            self._write(ledger)


def _native_state_hash(state):
    # Same physical snapshot digest as backend._state_hash; no backend/Genesis import.
    digest = hashlib.sha256()
    for key in sorted(set(state) - {"state_hash"}):
        value = np.ascontiguousarray(state[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _state_identity(state):
    return {
        key: state.get(key)
        for key in (
            "schema_version",
            "engine",
            "engine_version",
            "runtime_identity_json",
            "substeps",
            "mesh_hash",
            "config_json",
            "config_hash",
            "calibration_json",
        )
    }


def _physical_content(state):
    # Exclude seed/reset metadata so derivatives cannot cross splits under a new label.
    return _digest({key: state[key] for key in ("positions", "velocities", "free", "mesh_hash")})


def _fixture(manifest, fixture_id):
    for fixture in manifest["fixtures"]:
        if fixture["fixture_id"] == fixture_id:
            return fixture
    raise ValueError(f"unknown fixture: {fixture_id}")


def _verify_refs(root, value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            if file_sha256(relative_path(root, value["path"])) != value["sha256"]:
                raise ValueError(f"artifact hash mismatch: {value['path']}")
            if value["path"].endswith(".json"):
                _verify_refs(root, load_json(root, value["path"]))
        for item in value.values():
            _verify_refs(root, item)
    elif isinstance(value, list):
        for item in value:
            _verify_refs(root, item)


def _accept_state(state, config, coverage, pairs):
    p, v = (np.asarray(state[key]) for key in ("positions", "velocities"))
    if p.ndim != 2 or p.shape[1] != 3 or v.shape != p.shape:
        raise ValueError("invalid state geometry")
    finite = bool(np.isfinite(p).all() and np.isfinite(v).all())
    metadata = json.loads(state["reset_metadata_json"])
    edges = np.asarray(state["edges"], dtype=int)
    rest = np.asarray(state["rest_positions"])
    rest_lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    lengths = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
    if not np.isfinite(rest_lengths).all() or np.any(rest_lengths <= 0):
        raise ValueError("invalid rest edge geometry")
    strain = float(np.max(lengths / rest_lengths - 1))
    checks = {
        "finite_state": finite,
        "reset_settled": metadata.get("status") == "valid"
        and metadata.get("stable_seconds", 0) >= 0.5
        and metadata.get("final_rms_speed_m_s", math.inf) < 0.01,
        "table_contained": finite
        and bool((np.abs(p[:, :2]) <= config["table_size_m"] / 2).all())
        and float(p[:, 2].min()) >= -0.01,
        "coverage_in_range": math.isfinite(coverage) and 0.15 <= coverage <= 0.85,
        "strain_guard": math.isfinite(strain) and strain <= 0.10 + 1e-6,
        "visible_feasible_pair": bool(pairs),
        "released": bool(np.asarray(state["free"], dtype=bool).all()),
    }
    return checks, strain


def _validate_state(state, identity, snapshot_identity=None):
    if (
        state.get("schema_version") != 2
        or state.get("engine") != "genesis"
        or state.get("engine_version") != "1.4.4"
    ):
        raise ValueError("fresh Genesis 1.4.4 schema-2 state required")
    if state.get("state_hash") != _native_state_hash(state):
        raise ValueError("native state content hash mismatch")
    if json.loads(state["config_json"]) != identity["physics_config"]:
        raise ValueError("state physics config differs from frozen identity")
    if json.loads(state["runtime_identity_json"]) != identity["runtime_identity"]:
        raise ValueError("state runtime/numerical identity differs")
    if snapshot_identity is not None and _state_identity(state) != snapshot_identity:
        raise ValueError("state scene/mesh/numerical identity differs")


def prepare_benchmark(
    env,
    output_dir,
    *,
    appearance=None,
    max_wall_seconds=WALL_LIMIT,
    per_reset_wall_seconds=900,
    pair_finder=None,
):
    """Prepare folds then the first six qualifying drops per split, before any action.

    Every scheduled seed is reserved/journaled before reset. Resume never replays an
    uncertain reset. Failed snapshots remain support evidence, including nonfinite data.
    A pool with fewer than six stops preparation with a frozen ``pool_limitation``.
    """
    from .collection import _deadline
    from .selection import NoValidPairError, feasible_candidates

    if not 0 < max_wall_seconds <= WALL_LIMIT or not 0 < per_reset_wall_seconds <= WALL_LIMIT:
        raise ValueError("invalid preparation wall budget")
    identity = benchmark_identity(env, appearance=appearance)
    if pair_finder is None:
        pair_finder = feasible_candidates
    with _locked(output_dir, ".benchmark-prepare.lock") as root:
        if (root / MANIFEST).exists():
            return load_benchmark(root, identity=identity)
        ledger = BudgetLedger(root)
        if (root / "benchmark-preparation.json").exists():
            progress = load_json(root, "benchmark-preparation.json")
            if progress["identity"] != identity:
                raise ValueError("preparation identity changed")
        else:
            progress = {"schema_version": 1, "identity": identity, "attempts": [], "files": {}}
        # Reconcile the tiny reserve->journal window without retrying its reset.
        for entry in ledger.status()["pending"]:
            if not entry["category"].startswith("prepare_"):
                raise RuntimeError("unresolved non-preparation simulator reservation")
            attempt = next(
                (a for a in progress["attempts"] if a["reservation"] == entry["token"]), None
            )
            if attempt is None:
                m = entry["metadata"]
                attempt = {
                    "fixture_id": f"{m['split']}-{m['kind']}-{m['seed']}",
                    "seed": m["seed"],
                    "kind": m["kind"],
                    "split": m["split"],
                    "reset_group_id": f"{m['kind']}-{m['seed']}",
                    "cloth_id": CLOTH_ID,
                    "reservation": entry["token"],
                    "status": "preparing",
                }
                progress["attempts"].append(attempt)
            ledger.abandon(entry["token"], result={"reason": "interrupted_preparation"})
            if attempt["status"] == "preparing":
                attempt.update(status="rejected", accepted=False, reason="interrupted_preparation")
        for attempt in progress["attempts"]:
            if attempt["status"] == "preparing":
                attempt.update(status="rejected", accepted=False, reason="interrupted_preparation")
        _verify_refs(root, list(progress["files"].values()))
        deadline = time.monotonic() + max_wall_seconds
        stop_reason = "completed"

        def save():
            atomic_write(root, "benchmark-preparation.json", json_bytes(progress))

        def put(reference, data):
            if relative_path(root, reference).exists():
                raise FileExistsError(reference)
            atomic_write(root, reference, data)
            ref = {"path": reference, "sha256": sha256(data)}
            progress["files"][reference] = ref
            return ref

        save()
        schedules = [("development", "fold", (11, 17, 23))]
        schedules += [
            ("development", "drop", range(1000, 1012)),
            ("heldout", "drop", range(2000, 2012)),
        ]
        snapshot_identity = next(
            (a["snapshot_identity"] for a in progress["attempts"] if a.get("accepted")), None
        )
        for split, kind, seeds in schedules:
            for seed in seeds:
                accepted = [
                    a
                    for a in progress["attempts"]
                    if a["split"] == split and a["kind"] == kind and a.get("accepted")
                ]
                if kind == "drop" and len(accepted) >= 6:
                    break
                if any(a["seed"] == seed and a["kind"] == kind for a in progress["attempts"]):
                    continue
                allowance = min(
                    per_reset_wall_seconds,
                    deadline - time.monotonic(),
                    ledger.status()["remaining_wall_seconds"],
                )
                if allowance <= 0:
                    stop_reason = "wall_budget"
                    break
                token = ledger.reserve(
                    "prepare_" + kind,
                    wall_seconds=allowance,
                    key=f"prepare:{kind}:{seed}",
                    metadata={"split": split, "kind": kind, "seed": seed},
                )
                attempt = {
                    "fixture_id": f"{split}-{kind}-{seed}",
                    "seed": seed,
                    "split": split,
                    "kind": kind,
                    "reset_group_id": f"{kind}-{seed}",
                    "cloth_id": CLOTH_ID,
                    "status": "preparing",
                    "accepted": False,
                    "reservation": token,
                }
                progress["attempts"].append(attempt)
                save()
                started = time.monotonic()
                state = None
                try:
                    with _deadline(started + allowance):
                        observation = env.reset(seed=seed, kind=kind)
                        state = env.snapshot()
                        _validate_state(state, identity, snapshot_identity)
                        coverage = float(env.coverage()[0])
                        try:
                            pairs = pair_finder(observation, env.config, max_pairs=1)
                        except NoValidPairError:
                            pairs = []
                        checks, strain = _accept_state(
                            state, identity["physics_config"], coverage, pairs
                        )
                    accepted = all(checks.values())
                    attempt.update(
                        accepted=accepted,
                        checks=checks,
                        coverage=coverage,
                        max_edge_strain=strain,
                        reason="accepted" if accepted else "pre_action_rejected",
                        snapshot_identity=_state_identity(state),
                        state_content_sha256=_physical_content(state),
                        state_sha256=_digest(state),
                    )
                    if accepted and snapshot_identity is None:
                        snapshot_identity = _state_identity(state)
                except Exception as exc:  # Every failed reset is evidence, never a flat fallback.
                    attempt["reason"] = f"{type(exc).__name__}: {exc}"
                    if state is None:
                        try:
                            with _deadline(started + allowance):
                                state = env.snapshot()
                        except Exception as capture_error:
                            attempt["snapshot_error"] = (
                                f"{type(capture_error).__name__}: {capture_error}"
                            )
                finally:
                    elapsed = time.monotonic() - started
                    # Timed-out calls consume the allowance; their overrun remains explicit.
                    attempt["wall_seconds"] = elapsed
                    if elapsed > allowance:
                        ledger.abandon(
                            token, result={"reason": "deadline", "observed_wall_seconds": elapsed}
                        )
                    else:
                        ledger.finish(
                            token, wall_seconds=elapsed, result={"reason": attempt["reason"]}
                        )
                    if state is not None:
                        attempt["state"] = write_bundle(put, state, f"fixtures/{kind}-{seed}")
                    attempt["status"] = "accepted" if attempt["accepted"] else "rejected"
                    save()
            if (
                kind == "fold"
                and sum(
                    a.get("accepted", False) for a in progress["attempts"] if a["kind"] == "fold"
                )
                < 3
            ):
                stop_reason = "fixture_limitation" if stop_reason == "completed" else stop_reason
                break
            if (
                kind == "drop"
                and sum(
                    a.get("accepted", False)
                    for a in progress["attempts"]
                    if a["split"] == split and a["kind"] == kind
                )
                < 6
            ):
                stop_reason = "pool_limitation" if stop_reason == "completed" else stop_reason
                break
            if stop_reason != "completed":
                break
        manifest = {
            "schema_version": 1,
            "milestone": 3,
            "frozen": True,
            "identity": identity,
            "identity_sha256": _digest(identity),
            "snapshot_identity": snapshot_identity,
            "cloth_id": CLOTH_ID,
            "scope": "single synthetic 0.50m square; folds are development-only and not independent generalization",
            "criteria": {
                "coverage_min": 0.15,
                "coverage_max": 0.85,
                "max_strain": 0.10,
                "per_pool": 6,
                "max_drop_attempts": 24,
            },
            "stop_reason": stop_reason,
            "complete": stop_reason == "completed",
            "stage_passed": stop_reason == "completed",
            "attempts": progress["attempts"],
            "fixtures": [a for a in progress["attempts"] if a.get("accepted")],
            "files": progress["files"],
        }
        _sealed_write(root, MANIFEST, manifest)
        return load_benchmark(root, identity=identity)


def load_benchmark(output_dir, *, identity=None):
    """Fail closed on changed scene, state, split, order, or supporting artifact."""
    root = Path(output_dir).resolve()
    manifest = _sealed_load(root, MANIFEST)
    if manifest.get("schema_version") != 1 or not manifest.get("frozen"):
        raise ValueError("unsupported/unfrozen benchmark manifest")
    if _digest(manifest["identity"]) != manifest["identity_sha256"]:
        raise ValueError("benchmark identity hash mismatch")
    if identity is not None and identity != manifest["identity"]:
        raise ValueError("frozen benchmark configuration mismatch")
    _verify_refs(root, list(manifest["files"].values()))
    fixtures = manifest["fixtures"]
    if fixtures != [a for a in manifest["attempts"] if a.get("accepted")]:
        raise ValueError("fixture selection differs from preparation journal")
    seen_content = {}
    seen_groups = set()
    for split, kind, seeds in (
        ("development", "fold", [11, 17, 23]),
        ("development", "drop", list(range(1000, 1012))),
        ("heldout", "drop", list(range(2000, 2012))),
    ):
        attempts = [a for a in manifest["attempts"] if a["split"] == split and a["kind"] == kind]
        if [a["seed"] for a in attempts] != seeds[: len(attempts)]:
            raise ValueError("preparation seed order changed")
        qualifying = [a for a in attempts if a.get("accepted")]
        if kind == "drop" and (
            len(qualifying) > 6
            or (qualifying and attempts[-1] != qualifying[-1] and len(qualifying) == 6)
        ):
            raise ValueError("first-six pool selection violated")
        if manifest["stop_reason"] == "completed" and len(qualifying) != (
            3 if kind == "fold" else 6
        ):
            raise ValueError("completed benchmark has missing fixtures")
    if len(manifest["attempts"]) > 27:
        raise ValueError("preparation attempt limit exceeded")
    for fixture in fixtures:
        state = read_bundle(root, fixture["state"]["path"])
        _validate_state(state, manifest["identity"], manifest["snapshot_identity"])
        if (
            _digest(state) != fixture["state_sha256"]
            or _physical_content(state) != fixture["state_content_sha256"]
        ):
            raise ValueError("frozen state hash mismatch")
        checks, strain = _accept_state(
            state,
            manifest["identity"]["physics_config"],
            fixture["coverage"],
            fixture["checks"]["visible_feasible_pair"],
        )
        if (
            not all(checks.values())
            or checks != fixture["checks"]
            or strain != fixture["max_edge_strain"]
        ):
            raise ValueError("fixture pre-action qualification mismatch")
        group = fixture["reset_group_id"]
        if group in seen_groups:
            raise ValueError("duplicate reset group")
        seen_groups.add(group)
        old_split = seen_content.setdefault(fixture["state_content_sha256"], fixture["split"])
        if old_split != fixture["split"]:
            raise ValueError("derivative state leakage across splits")
        if fixture["cloth_id"] != CLOTH_ID or (
            fixture["kind"] == "fold" and fixture["split"] != "development"
        ):
            raise ValueError("cloth/split identity changed")
    return manifest


def load_fixture(output_dir, fixture_id, *, identity=None):
    manifest = load_benchmark(output_dir, identity=identity)
    return read_bundle(Path(output_dir).resolve(), _fixture(manifest, fixture_id)["state"]["path"])


def _check_qualification(root, record, manifest):
    if record["manifest_sha256"] != file_sha256(root / MANIFEST):
        raise ValueError("qualification manifest hash mismatch")
    if record["benchmark_identity"] != manifest["identity"]:
        raise ValueError("qualification configuration mismatch")
    if not isinstance(record["passed"], bool):
        raise ValueError("qualification needs explicit passed boolean")
    for key in ("laydown", "fling"):
        if not isinstance(record["primitive_config"].get(key), dict):
            raise ValueError("qualification must freeze both primitive configurations")
    if not isinstance(record["selection_config"], dict):
        raise ValueError("qualification must freeze selector configuration")
    for category in ("laydown_runs", "fling_runs", "continuity"):
        if not isinstance(record.get(category), list):
            raise ValueError(f"qualification needs {category} evidence list")
        if record["passed"] and not record[category]:
            raise ValueError("passing qualification requires all evidence categories")
        for item in record[category]:
            fixture = _fixture(manifest, item["fixture_id"])
            if fixture["split"] != "development":
                raise ValueError("qualification cannot use heldout")
            ref = item["diagnostic_ref"]
            _verify_refs(root, ref)
    _verify_refs(root, record)


def record_qualification(output_dir, record, *, identity=None):
    """Freeze coordinator-recomputed qualification and settings before pilot/heldout.

    ``passed`` is the coordinator's physical gate conclusion, not inferred from
    infrastructure success. Every diagnostic source is hashed and development-only.
    """
    with _locked(output_dir, ".benchmark-qualification.lock") as root:
        manifest = load_benchmark(root, identity=identity)
        if not manifest["stage_passed"]:
            raise ValueError("incomplete fixture preparation cannot qualify")
        record = copy.deepcopy(record)
        _check_qualification(root, record, manifest)
        ledger = BudgetLedger(root).status()
        if ledger["pending"] or ledger["pilot_episodes"]:
            raise ValueError("cannot alter qualification during/after pilot")
        _sealed_write(root, "benchmark-qualification.json", record)
        return record


def load_qualification(output_dir):
    root = Path(output_dir).resolve()
    manifest = load_benchmark(root)
    record = _sealed_load(root, "benchmark-qualification.json")
    _check_qualification(root, record, manifest)
    return record
