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
        tokens = set()
        for entry in entries:
            reserved = entry.get("reserved_wall_seconds")
            charged = entry.get("wall_seconds", reserved)
            if (
                entry.get("category") not in LIMITS
                or entry.get("status") not in {"reserved", "finished", "abandoned"}
                or not isinstance(reserved, (int, float))
                or not math.isfinite(reserved)
                or reserved <= 0
                or not isinstance(charged, (int, float))
                or not math.isfinite(charged)
                or charged < 0
                or entry["token"] in tokens
            ):
                raise ValueError("invalid benchmark ledger reservation")
            tokens.add(entry["token"])
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
            "entries": copy.deepcopy(entries),
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
                    if accepted and any(
                        a.get("accepted")
                        and a["split"] != split
                        and a.get("state_content_sha256") == _physical_content(state)
                        for a in progress["attempts"]
                    ):
                        raise ValueError("derivative state leakage across splits")
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
                except BaseException as exc:  # Persist interruption without retrying its reset.
                    attempt["reason"] = f"{type(exc).__name__}: {exc}"
                    attempt["interrupted"] = not isinstance(exc, Exception)
                    if state is None:
                        try:
                            with _deadline(started + allowance):
                                state = env.snapshot()
                        except Exception as capture_error:
                            attempt["snapshot_error"] = (
                                f"{type(capture_error).__name__}: {capture_error}"
                            )
                    if attempt["interrupted"]:
                        raise
                finally:
                    elapsed = time.monotonic() - started
                    # Timed-out calls consume the allowance; their overrun remains explicit.
                    attempt["wall_seconds"] = elapsed
                    if elapsed > allowance or attempt.get("interrupted"):
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
    if manifest.get("complete") != (manifest["stop_reason"] == "completed") or manifest.get(
        "stage_passed"
    ) != manifest.get("complete"):
        raise ValueError("benchmark completion flags mismatch")
    allowed_order = [("development", "fold", seed) for seed in (11, 17, 23)]
    allowed_order += [("development", "drop", seed) for seed in range(1000, 1012)]
    allowed_order += [("heldout", "drop", seed) for seed in range(2000, 2012)]
    ordered = [(a["split"], a["kind"], a["seed"]) for a in manifest["attempts"]]
    if any(item not in allowed_order for item in ordered) or len(ordered) != len(set(ordered)):
        raise ValueError("unknown or repeated preparation seed/split")
    if ordered != sorted(ordered, key=allowed_order.index):
        raise ValueError("preparation split order changed")
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
        if (
            fixture["fixture_id"] != f"{fixture['split']}-{fixture['kind']}-{fixture['seed']}"
            or fixture["reset_group_id"] != f"{fixture['kind']}-{fixture['seed']}"
        ):
            raise ValueError("fixture/reset-group identity changed")
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
            diagnostic = load_json(root, ref["path"])
            provenance = diagnostic.get("provenance")
            if provenance is not None:
                for key, value in manifest["identity"].items():
                    if provenance.get(key) != value:
                        raise ValueError(f"qualification source configuration mismatch: {key}")
                source_hash = provenance.get("source_reset_sha256")
                if source_hash != fixture["state_sha256"] and (
                    source_hash is not None or record["passed"]
                ):
                    raise ValueError("qualification source frozen state hash mismatch")
            if (
                "primitive_config" in item
                and "config" in diagnostic
                and item["primitive_config"] != diagnostic["config"]
            ):
                raise ValueError("qualification evidence primitive configuration mismatch")
            if (
                "pixels" in item
                and "fixed_pixels" in diagnostic
                and _digest(item["pixels"]) != _digest(diagnostic["fixed_pixels"])
            ):
                raise ValueError("qualification fixed pair mismatch")
            if "stage" in item and item["stage"] not in {"initial", "revision"}:
                raise ValueError("unknown qualification tuning stage")
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


def aggregate_episode(
    rows, *, initial_coverage, evidence=None, end_reason=None, complete=False, max_actions=5
):
    """Initial+5 confirmed-settled curve with invalid terminal geometry kept separate.

    ``evidence`` maps transition IDs to the existing writer evidence payload. A
    valid action advances coverage only with an explicit settled diagnostic. Missing,
    invalid and unconfirmed steps carry the last confirmed settled coverage forward.
    Signed improvements are never clipped. Incomplete/failure flags remain visible.
    """
    if not math.isfinite(initial_coverage):
        raise ValueError("finite initial settled coverage required")
    if len(rows) > max_actions or [r["step_id"] for r in rows] != list(range(len(rows))):
        raise ValueError("pilot step order/action bound violated")
    evidence = evidence or {}
    curve = [float(initial_coverage)]
    observed_invalid = []
    unconfirmed = []
    recorded = {"attempts": 0, "wall_seconds": 0.0, "simulated_seconds": 0.0}
    unrecorded = dict(recorded)
    valid = 0
    for row in rows:
        support = evidence.get(row["transition_id"], {})
        diagnostics = support.get("diagnostics", {})
        recording = support.get("evidence", {}).get("recording")
        timing = recorded if recording is not None else unrecorded
        timing["attempts"] += 1
        for key in ("wall_seconds", "simulated_seconds"):
            value = row.get(key)
            if value is not None and math.isfinite(value):
                timing[key] += value
        value = row.get("coverage_after")
        finite = value is not None and math.isfinite(value)
        if row["execution_valid"]:
            valid += 1
            if diagnostics.get("settled") is True and finite:
                curve.append(float(value))
            else:
                curve.append(curve[-1])
                unconfirmed.append(row["step_id"])
        else:
            curve.append(curve[-1])
            observed_invalid.append(
                {
                    "step_id": row["step_id"],
                    "observed_coverage": value if finite else None,
                    "settled": diagnostics.get("settled"),
                    "reason": row["reason"],
                    "terminal_observation_valid": diagnostics.get("terminal_observation_valid"),
                }
            )
    curve.extend([curve[-1]] * (max_actions + 1 - len(curve)))
    final = curve[-1]
    return {
        "curve": curve,
        "initial_coverage": initial_coverage,
        "final_coverage": final,
        "final_improvement": final - initial_coverage,
        "success_coverage_ge_0_95": final >= 0.95,
        "attempts": len(rows),
        "valid_attempts": valid,
        "invalid_attempts": len(rows) - valid,
        "valid_attempt_fraction": valid / len(rows) if rows else None,
        "invalid_attempt_fraction": (len(rows) - valid) / len(rows) if rows else None,
        "no_pair": end_reason == "no_valid_pair",
        "end_reason": end_reason,
        "complete": bool(complete and not unconfirmed),
        "missing": False,
        "unconfirmed_valid_steps": unconfirmed,
        "observed_invalid_terminal_geometry": observed_invalid,
        "timing": {"recorded": recorded, "unrecorded": unrecorded},
    }


def _pilot_binding(dataset, episode):
    run = next(r for r in dataset["runs"] if r["run_id"] == episode["run_id"])
    episode_meta = episode.get("metadata", {})
    run_meta = run["metadata"]
    binding = episode_meta.get("benchmark") or run_meta.get("episode_metadata", {}).get("benchmark")
    # Also accept the original documented direct-run convention for older local callers.
    binding = binding or run_meta.get("benchmark")
    if not isinstance(binding, dict):
        raise ValueError("pilot dataset is missing benchmark identity metadata")
    initial_settled = episode_meta.get("initial_settled")
    if initial_settled is None:
        initial_settled = run_meta.get("episode_metadata", {}).get("initial_settled")
    return binding, run_meta, initial_settled


def _read_pilot_episode(root, dataset_root, dataset, episode, rows, manifest, qualification):
    binding, run_meta, initial_settled = _pilot_binding(dataset, episode)
    fixture = _fixture(manifest, binding["fixture_id"])
    selector = binding["selector"]
    if fixture["kind"] != "drop" or selector not in SELECTORS:
        raise ValueError("pilot dataset has an undeclared fixture/selector")
    if binding["manifest_sha256"] != file_sha256(root / MANIFEST):
        raise ValueError("pilot frozen manifest mismatch")
    if qualification is None or binding["qualification_sha256"] != _digest(qualification):
        raise ValueError("pilot frozen qualification mismatch")
    if not qualification["passed"]:
        raise ValueError("pilot cannot follow failed qualification")
    for key in ("split", "reset_group_id", "cloth_id"):
        if episode[key] != fixture[key]:
            raise ValueError(f"pilot {key} differs from frozen reset")
    reset = read_bundle(dataset_root, episode["reset"]["path"])
    source = reset.get("reset_source")
    if source is None or _digest(source) != fixture["state_sha256"]:
        raise ValueError("pilot is not restored from exact paired frozen state")
    observed_reset = reset.get("state")
    if observed_reset is not None and not episode.get("reset_error"):
        _validate_state(observed_reset, manifest["identity"], manifest["snapshot_identity"])
        if _physical_content(observed_reset) != fixture["state_content_sha256"]:
            raise ValueError("observed pilot reset differs from exact frozen state")
    elif not episode.get("reset_error"):
        raise ValueError("observed pilot reset state missing")
    if reset["config"] != manifest["identity"]["physics_config"]:
        raise ValueError("pilot physics configuration changed")
    provenance = run_meta.get("configuration_provenance", {})
    for key, value in manifest["identity"].items():
        if provenance.get(key) != value:
            raise ValueError(f"pilot frozen configuration changed: {key}")
    # Pilot uses the qualified laydown configuration; settings cannot be retuned on heldout.
    if run_meta.get("primitive_config") != qualification["primitive_config"]["laydown"]:
        raise ValueError("pilot primitive configuration changed after qualification")
    if (
        "selection_config" in run_meta
        and run_meta["selection_config"] != qualification["selection_config"]
    ):
        raise ValueError("pilot selector configuration changed after qualification")
    if (
        "selector_config" in run_meta
        and run_meta["selector_config"] != qualification["selection_config"]
    ):
        raise ValueError("pilot selector configuration changed after qualification")
    if rows and initial_settled is not True:
        raise ValueError("pilot initial settled confirmation missing")
    if rows and (
        rows[0].get("coverage_before") is None
        or not math.isclose(
            rows[0]["coverage_before"], fixture["coverage"], rel_tol=0, abs_tol=1e-8
        )
    ):
        raise ValueError("pilot initial coverage differs from frozen state")
    for row in rows:
        allowed_policies = {selector}
        if selector == "geometry_guided":
            allowed_policies.add("geometry_guided_privileged_depth")
        if row["policy"] not in allowed_policies or row["primitive"] != "laydown":
            raise ValueError("pilot transition selector/primitive differs from frozen protocol")
    support = {r["transition_id"]: read_bundle(dataset_root, r["evidence_path"]) for r in rows}
    complete = episode["status"] == "closed" and episode["end_reason"] in {
        "max_actions",
        "no_valid_pair",
        "execution_invalid",
        "environment_terminated",
        "settled_coverage_threshold",
        "initially_unfolded",
    }
    if episode.get("reset_error") or episode["end_reason"] in {
        "interrupted",
        "max_wall_seconds",
        "reset_failed",
        "reset_timeout",
        "selection_error",
        "selection_timeout",
    }:
        complete = False
    result = aggregate_episode(
        rows,
        initial_coverage=fixture["coverage"],
        evidence=support,
        end_reason=episode["end_reason"],
        complete=complete,
    )
    result.update(
        fixture_id=fixture["fixture_id"],
        selector=selector,
        split=fixture["split"],
        reset_group_id=fixture["reset_group_id"],
        dataset=dataset_root.relative_to(root).as_posix(),
        episode_id=episode["episode_id"],
        state_content_sha256=fixture["state_content_sha256"],
    )
    result["rgb"] = []
    for label, observation in (
        ("initial", episode.get("initial_observation")),
        ("terminal", episode.get("last_observation")),
    ):
        if observation:
            ref = observation["rgb"]["path"]
            result["rgb"].append(
                {
                    "label": label,
                    "relative_file": relative_path(dataset_root, ref).relative_to(root).as_posix(),
                }
            )
    videos = []

    def find_videos(value):
        if isinstance(value, dict):
            reference = value.get("relative_file")
            if isinstance(reference, str) and Path(reference).suffix.lower() in {".mp4", ".webm"}:
                path = relative_path(dataset_root, reference)
                if not path.is_file():
                    raise ValueError(f"missing native recording: {reference}")
                if value.get("sha256") and file_sha256(path) != value["sha256"]:
                    raise ValueError("native recording hash mismatch")
                videos.append(path.relative_to(root).as_posix())
            for child in value.values():
                find_videos(child)
        elif isinstance(value, list):
            for child in value:
                find_videos(child)

    for item in support.values():
        find_videos(item.get("evidence", {}).get("recording", {}))
    result["native_videos"] = sorted(set(videos))
    return result


def _summarize_selector(episodes):
    n = len(episodes)
    attempts = sum(e["attempts"] for e in episodes)
    valid = sum(e["valid_attempts"] for e in episodes)
    observed = sum(not e["missing"] for e in episodes)
    return {
        "groups": n,
        "observed_episodes": observed,
        "curve_status": "observed_with_missing_baselines"
        if observed and observed < n
        else "observed"
        if observed
        else "not_evaluated",
        "complete_episodes": sum(e["complete"] for e in episodes),
        "mean_initial_coverage": sum(e["initial_coverage"] for e in episodes) / n if n else None,
        "mean_final_coverage": sum(e["final_coverage"] for e in episodes) / n
        if n and observed
        else None,
        "mean_final_improvement": sum(e["final_improvement"] for e in episodes) / n
        if n and observed
        else None,
        "positive_final_improvement_groups": sum(e["final_improvement"] > 0 for e in episodes),
        "success_fraction_coverage_ge_0_95": sum(e["success_coverage_ge_0_95"] for e in episodes)
        / n
        if n
        else None,
        "attempts": attempts,
        "valid_attempts": valid,
        "invalid_attempts": attempts - valid,
        "valid_attempt_fraction": valid / attempts if attempts else None,
        "invalid_attempt_fraction": (attempts - valid) / attempts if attempts else None,
        "no_pair_episodes": sum(e["no_pair"] for e in episodes),
        "no_pair_episode_fraction": sum(e["no_pair"] for e in episodes) / n
        if n and observed
        else None,
        "mean_curve": [
            sum(e["curve"][i] if e["curve"] else e["initial_coverage"] for e in episodes) / n
            for i in range(6)
        ]
        if n and observed
        else [],
        "timing": {
            mode: {
                key: sum(e["timing"][mode][key] for e in episodes)
                for key in ("attempts", "wall_seconds", "simulated_seconds")
            }
            for mode in ("recorded", "unrecorded")
        },
    }


def report_benchmark(output_dir):
    """Write a compact offline JSON/HTML review, keeping missing/failure denominators.

    Reads only the existing transition schema, metadata and referenced artifacts.
    Does not import Genesis/Torch, restore a state, or execute/tune a selector.
    """
    from .data import read_transitions, verify_dataset

    root = Path(output_dir).resolve()
    manifest = load_benchmark(root)
    qualification = (
        load_qualification(root) if (root / "benchmark-qualification.json").exists() else None
    )
    by_pair = {}
    integrity_errors = []
    pilot_root = root / "pilot"
    paths = sorted(pilot_root.rglob("manifest.json")) if pilot_root.exists() else []
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("pilot path escapes benchmark root")
        directory = path.parent
        dataset = load_json(directory, "manifest.json")
        verification = verify_dataset(directory)
        # Structural failures are retained as incomplete report evidence; content mismatches
        # and identity tampering are always rejected instead of silently dropping the run.
        fatal = [
            e
            for e in verification["errors"]
            if "unfinished" not in e
            and "still open" not in e
            and "in-flight" not in e
            and "terminal reason" not in e
            and "terminal/truncation" not in e
        ]
        if fatal:
            raise ValueError("pilot dataset integrity: " + "; ".join(fatal))
        integrity_errors.extend(
            f"{directory.relative_to(root)}: {e}" for e in verification["errors"]
        )
        rows = read_transitions(directory).to_pylist()
        if not dataset["episodes"]:
            integrity_errors.append(f"{directory.relative_to(root)}: dataset has no episode")
        for episode in dataset["episodes"]:
            selected = [r for r in rows if r["episode_id"] == episode["episode_id"]]
            result = _read_pilot_episode(
                root, directory, dataset, episode, selected, manifest, qualification
            )
            pair = (result["fixture_id"], result["selector"])
            if pair in by_pair:
                raise ValueError("duplicate/rerun pilot fixture-selector episode")
            by_pair[pair] = result
    ledger = BudgetLedger(root).status()
    claims = {(e["fixture_id"], e["selector"]): e for e in ledger["pilot_episodes"]}
    episodes = []
    for fixture in manifest["fixtures"]:
        if fixture["kind"] != "drop":
            continue
        for selector in SELECTORS:
            pair = (fixture["fixture_id"], selector)
            if pair in by_pair:
                result = by_pair[pair]
                if pair not in claims:
                    integrity_errors.append(f"{pair}: missing durable pilot episode claim")
                    result["complete"] = False
                elif claims[pair]["status"] != "finished":
                    result["complete"] = False
                result["ledger_claim"] = claims.get(pair)
                if pair in claims:
                    claim = claims[pair]
                    if (
                        claim["qualification_sha256"] != _digest(qualification)
                        or claim["state_content_sha256"] != fixture["state_content_sha256"]
                    ):
                        raise ValueError("pilot ledger frozen identity mismatch")
                    reservations = [
                        entry
                        for entry in ledger["entries"]
                        if entry["category"] == "pilot"
                        and entry["metadata"].get("episode_token") == claim["token"]
                    ]
                    if len(reservations) != result["attempts"]:
                        integrity_errors.append(f"{pair}: dataset/reserved action count mismatch")
                        result["complete"] = False
            else:
                result = aggregate_episode([], initial_coverage=fixture["coverage"])
                result.update(
                    fixture_id=fixture["fixture_id"],
                    selector=selector,
                    split=fixture["split"],
                    reset_group_id=fixture["reset_group_id"],
                    missing=True,
                    curve=[],
                    curve_status="missing_episode",
                    end_reason="missing_episode",
                    rgb=[],
                    native_videos=[],
                    ledger_claim=claims.get(pair),
                )
            episodes.append(result)
    splits = {}
    for split in ("development", "heldout"):
        summaries = {
            selector: _summarize_selector(
                [e for e in episodes if e["split"] == split and e["selector"] == selector]
            )
            for selector in SELECTORS
        }
        differences = []
        for fixture in manifest["fixtures"]:
            if fixture["kind"] != "drop" or fixture["split"] != split:
                continue
            paired = [
                next(
                    e
                    for e in episodes
                    if e["fixture_id"] == fixture["fixture_id"] and e["selector"] == selector
                )
                for selector in SELECTORS
            ]
            differences.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "geometry_minus_uniform_final_coverage": (
                        paired[1]["final_coverage"] - paired[0]["final_coverage"]
                    )
                    if all(not e["missing"] for e in paired)
                    else None,
                    "both_complete": all(e["complete"] for e in paired),
                    "both_observed": all(not e["missing"] for e in paired),
                }
            )
        summaries["paired_differences"] = differences
        summaries["mean_paired_final_difference"] = (
            sum(
                p["geometry_minus_uniform_final_coverage"]
                for p in differences
                if p["geometry_minus_uniform_final_coverage"] is not None
            )
            / sum(p["geometry_minus_uniform_final_coverage"] is not None for p in differences)
            if any(p["geometry_minus_uniform_final_coverage"] is not None for p in differences)
            else None
        )
        splits[split] = summaries
    geometry = splits["heldout"]["geometry_guided"]
    heldout_all = [e for e in episodes if e["split"] == "heldout"]
    gate_checks = {
        "six_heldout_groups": geometry["groups"] == 6,
        "all_paired_episodes_complete": len(heldout_all) == 12
        and all(e["complete"] for e in heldout_all),
        "mean_final_improvement_ge_0_10": geometry["mean_final_improvement"] is not None
        and geometry["mean_final_improvement"] >= 0.10 - 1e-12,
        "positive_improvement_ge_4_of_6": geometry["positive_final_improvement_groups"] >= 4,
        "valid_attempt_fraction_ge_0_80": geometry["valid_attempt_fraction"] is not None
        and geometry["valid_attempt_fraction"] >= 0.80,
        "evidence_integrity": not integrity_errors and not ledger["pending"],
        "fixture_preparation_passed": manifest["stage_passed"],
        "qualification_passed": bool(qualification and qualification["passed"]),
    }
    heldout_evaluated = any(not e["missing"] for e in heldout_all)
    pilot_evaluated = any(not e["missing"] for e in episodes)
    if not heldout_evaluated:
        for key in (
            "mean_final_improvement_ge_0_10",
            "positive_improvement_ge_4_of_6",
            "valid_attempt_fraction_ge_0_80",
        ):
            gate_checks[key] = None
    gate_status = (
        "not_evaluated"
        if not heldout_evaluated
        else "incomplete"
        if not gate_checks["all_paired_episodes_complete"]
        else "passed"
        if all(gate_checks.values())
        else "not_passed"
    )
    preparation = {
        "stop_reason": manifest["stop_reason"],
        "complete": manifest["complete"],
        "stage_passed": manifest["stage_passed"],
        "counts": {},
        "attempts": [],
    }
    for split in ("development", "heldout"):
        for kind in ("fold", "drop"):
            attempts = [
                a for a in manifest["attempts"] if a["split"] == split and a["kind"] == kind
            ]
            if attempts or kind == "drop":
                accepted = sum(a.get("accepted", False) for a in attempts)
                preparation["counts"][f"{split}_{kind}"] = {
                    "attempted": len(attempts),
                    "accepted": accepted,
                    "rejected": len(attempts) - accepted,
                    "required": 3 if kind == "fold" else 6,
                }
    for attempt in manifest["attempts"]:
        item = copy.deepcopy(attempt)
        item["failed_checks"] = [k for k, value in attempt.get("checks", {}).items() if not value]
        if attempt.get("state"):
            document = load_json(root, attempt["state"]["path"])
            item["state_arrays"] = document.get("arrays")
        preparation["attempts"].append(item)
    report = {
        "schema_version": 1,
        "simulation_only": True,
        "preparation": preparation,
        "stages": {
            "prepare": manifest["stop_reason"],
            "qualify": "not_evaluated"
            if qualification is None
            else "passed"
            if qualification["passed"]
            else "not_passed",
            "pilot": "not_evaluated"
            if not pilot_evaluated
            else "complete"
            if all(e["complete"] for e in episodes)
            else "incomplete",
            "heldout": gate_status,
        },
        "manifest_sha256": file_sha256(root / MANIFEST),
        "qualification_sha256": _digest(qualification) if qualification is not None else None,
        "scope": manifest["scope"],
        "selectors": {
            "uniform_valid": "visible feasible RGB candidates; privileged depth feasibility",
            "geometry_guided": "geometry-guided privileged-depth-assisted baseline",
        },
        "curve_definition": "initial+5; only explicitly confirmed settled valid terminal coverage advances; invalid/missing steps carry last confirmed settled coverage; negative gains retained",
        "success_definition": "final confirmed settled coverage >= 0.95; fraction includes missing/failure groups",
        "episodes": episodes,
        "splits": splits,
        "budget": ledger,
        "integrity_errors": integrity_errors,
        "heldout_geometry_gate": {
            "passed": all(gate_checks.values()) if heldout_evaluated else None,
            "evaluated": heldout_evaluated,
            "status": gate_status,
            "checks": gate_checks,
        },
        "offline_analysis_figures": [
            {"path": name, "sha256": file_sha256(root / name), "label": label}
            for name, label in (
                ("analysis/preparation-coverage.png", "Pre-action reset acceptance"),
                ("analysis/saved-geometry.png", "Saved geometry; offline plots, not native RGB"),
            )
            if (root / name).is_file()
        ],
    }
    atomic_write(root, "benchmark-report.json", json_bytes(report))
    atomic_write(root, "index.html", _review_html(report).encode())
    return report


def _review_html(report):
    from html import escape
    from urllib.parse import quote

    def esc(value):
        return escape(str(value), quote=True)

    def number(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"

    chunks = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Cloth-fling milestone 3 review</title>",
        "<style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;background:#fafafa;color:#18232c}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{text-align:left;padding:.5rem;border-bottom:1px solid #ccd4dc}figure{margin:1rem 0;padding:1rem;background:white;border:1px solid #ddd}img{max-width:220px}svg{max-width:100%;height:auto}.status{font-weight:bold}small{display:block}a{color:#174d9d}</style>",
        "<h1>Milestone 3 offline review</h1>",
        f"<p>{esc(report['scope'])}</p>",
        f"<p class='status'>Heldout geometry gate: {esc(report['heldout_geometry_gate']['status'].replace('_', ' ').upper())}</p>",
        "<p>Local simulation-only assisted attachment evidence. Geometry-guided selection uses privileged depth. Native recordings show grasp spheres when supplied by the coordinator recorder.</p>",
        f"<p>{esc(report['curve_definition'])}</p>",
        f"<p>{esc(report['success_definition'])}</p>",
    ]
    chunks.append(
        f"<h2>Reset preparation</h2><p class='status'>{esc(report['preparation']['stop_reason'])}. Qualification: {esc(report['stages']['qualify'])}; pilot: {esc(report['stages']['pilot'])}. Reset preparation limits do not establish failed action behavior.</p>"
    )
    chunks.append(
        "<table><tr><th>Pool</th><th>Attempts</th><th>Accepted / required</th><th>Rejected</th></tr>"
    )
    for pool, values in report["preparation"]["counts"].items():
        chunks.append(
            f"<tr><td>{esc(pool)}</td><td>{values['attempted']}</td><td>{values['accepted']} / {values['required']}</td><td>{values['rejected']}</td></tr>"
        )
    chunks.append(
        "</table><details open><summary>Every reset preparation and rejection</summary><table><tr><th>Fixture / split</th><th>Status</th><th>Coverage</th><th>Reason / failed pre-action checks</th><th>Saved state</th></tr>"
    )
    for attempt in report["preparation"]["attempts"]:
        links = []
        for label, ref in (("JSON", attempt.get("state")), ("NPZ", attempt.get("state_arrays"))):
            if ref:
                links.append(f"<a href='{esc(quote(ref['path'], safe='/'))}'>{label}</a>")
        reason = str(attempt.get("reason", ""))
        if attempt["failed_checks"]:
            reason += ": " + ", ".join(attempt["failed_checks"])
        chunks.append(
            f"<tr><td>{esc(attempt['fixture_id'])}</td><td>{esc(attempt['status'])}</td><td>{number(attempt.get('coverage'))}</td><td>{esc(reason)}</td><td>{' / '.join(links) if links else 'Snapshot unavailable'}</td></tr>"
        )
    chunks.append(
        "</table></details><p>Accepted and rejected state bundles are privileged execution/evaluation artifacts. They are not learned RGB observations.</p>"
    )
    for figure in report.get("offline_analysis_figures", []):
        chunks.append(
            f"<figure><img style='max-width:100%' src='{esc(quote(figure['path'], safe='/'))}' "
            f"alt='{esc(figure['label'])}'><figcaption>{esc(figure['label'])}</figcaption></figure>"
        )
    for split, summaries in report["splits"].items():
        chunks.append(
            f"<h2>{esc(split)}</h2><table><tr><th>Selector</th><th>Complete/groups</th><th>Final coverage</th><th>Gain</th><th>Success fraction</th><th>Invalid attempts</th><th>No pair</th></tr>"
        )
        for selector in SELECTORS:
            s = summaries[selector]
            chunks.append(
                f"<tr><td>{esc(selector)}</td><td>{s['complete_episodes']}/{s['groups']}</td><td>{number(s['mean_final_coverage'])}</td><td>{number(s['mean_final_improvement'])}</td><td>{number(s['success_fraction_coverage_ge_0_95'])}</td><td>{number(s['invalid_attempt_fraction'])}</td><td>{number(s['no_pair_episode_fraction'])}</td></tr>"
            )
        if not any(summaries[selector]["mean_curve"] for selector in SELECTORS):
            chunks.append(
                "</table><p>No pilot actions: coverage-versus-action curves are not evaluated.</p>"
            )
            continue
        chunks.append(
            "</table><svg viewBox='0 0 640 270' role='img' aria-label='Mean confirmed settled coverage versus action'>"
        )
        chunks.append("<rect width='640' height='270' fill='white'/>")
        for y in (0, 0.25, 0.5, 0.75, 1):
            py = 225 - y * 180
            chunks.append(
                f"<path d='M50 {py}H610' stroke='#ddd'/><text x='8' y='{py + 5}' font-size='13'>{y:.2f}</text>"
            )
        for i in range(6):
            chunks.append(f"<text x='{50 + i * 112}' y='248' font-size='13'>{i}</text>")
        chunks.append("<text x='265' y='266' font-size='13'>Action number</text>")
        for selector, color in zip(SELECTORS, ("#3369b2", "#b44524"), strict=True):
            curve = summaries[selector]["mean_curve"]
            points = " ".join(
                f"{50 + i * 112},{225 - value * 180}" for i, value in enumerate(curve)
            )
            chunks.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{points}'/>"
            )
        chunks.append("</svg><p>Blue: uniform_valid. Orange: geometry_guided.</p>")
        if not any(summaries[selector]["observed_episodes"] for selector in SELECTORS):
            chunks.append(
                "<p class='status'>NOT EVALUATED: no pilot episodes. Curves are missing; frozen reset coverage is preparation evidence only.</p>"
            )
        chunks.append(
            f"<p>Mean paired final coverage difference (geometry − uniform): {number(summaries['mean_paired_final_difference'])}</p>"
        )
    chunks.append("<h2>Representative episodes</h2>")
    categories = (
        ("Success", lambda e: e["success_coverage_ge_0_95"] and not e["missing"]),
        ("Negative gain", lambda e: e["final_improvement"] < 0),
        ("Invalid terminal", lambda e: bool(e["observed_invalid_terminal_geometry"])),
        ("Missing or incomplete", lambda e: e["missing"] or not e["complete"]),
    )
    for label, predicate in categories:
        episode = next((e for e in report["episodes"] if predicate(e)), None)
        chunks.append(f"<h3>{esc(label)}</h3>")
        if episode is None:
            chunks.append("<p>No observed example in these artifacts.</p>")
            continue
        chunks.append(
            f"<figure><figcaption>{esc(episode['fixture_id'])} / {esc(episode['selector'])}: final {number(episode['final_coverage'])}, gain {number(episode['final_improvement'])}; {esc(episode['end_reason'])}</figcaption>"
        )
        for rgb in episode.get("rgb", []):
            chunks.append(
                f"<img src='{esc(quote(rgb['relative_file'], safe='/'))}' alt='{esc(rgb['label'])} RGB'>"
            )
        for reference in episode.get("native_videos", []):
            chunks.append(
                f"<p><a href='{esc(quote(reference, safe='/'))}'>Native recording with grasp spheres: {esc(Path(reference).name)}</a></p>"
            )
        if episode["observed_invalid_terminal_geometry"]:
            chunks.append(
                f"<pre>{esc(json.dumps(episode['observed_invalid_terminal_geometry'], indent=2))}</pre>"
            )
        chunks.append("</figure>")
    chunks.append(
        "<h2>Timing by recording mode</h2><table><tr><th>Split / selector / capture</th><th>Attempts</th><th>Execution wall s</th><th>Simulated s</th></tr>"
    )
    for split, summaries in report["splits"].items():
        for selector in SELECTORS:
            for mode, values in summaries[selector]["timing"].items():
                chunks.append(
                    f"<tr><td>{esc(split)} / {esc(selector)} / {esc(mode)}</td><td>{values['attempts']}</td><td>{number(values['wall_seconds'])}</td><td>{number(values['simulated_seconds'])}</td></tr>"
                )
    chunks.append(
        "</table><p>Transition execution timings exclude reset/restore/selection and persistence overhead; the durable ledger separately charges whole simulator operation wall time.</p>"
    )
    chunks.append("<h2>Gate checks</h2><ul>")
    for key, passed in report["heldout_geometry_gate"]["checks"].items():
        chunks.append(f"<li>{esc(key)}: {esc('NOT EVALUATED' if passed is None else passed)}</li>")
    chunks.append("</ul>")
    if report["integrity_errors"]:
        chunks.append(
            "<h2>Incomplete evidence</h2><ul>"
            + "".join(f"<li>{esc(e)}</li>" for e in report["integrity_errors"])
            + "</ul>"
        )
    chunks.append("<p><a href='benchmark-report.json'>Full report and every episode</a></p></html>")
    return "\n".join(chunks)
