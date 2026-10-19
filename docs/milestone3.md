# Milestone 3: frozen robotless assisted-grasp benchmark

This milestone is bounded local simulation evidence for one synthetic 0.50 m square.
Genesis 1.4.4/native-v2 migration checks and historical Genesis 0.4.3 results do not
qualify the new primitives. Assisted attachments are not physical pinch grasps.
No training, publication, cloud, hardware, or broad collection is authorized.

## Preparation and integrity

`prepare_benchmark(env, output_dir, appearance=..., ...)` accepts an environment
already owned by an exclusive coordinator GPU lease. It never constructs a scene.
It prepares fresh schema-2 development folds 11/17/23, then tries development drop
seeds 1000..1011 followed by heldout 2000..2011, stopping each pool at its first six
qualifying resets. No post-action criterion selects a fixture. Acceptance requires
finite positions/velocities, reset settling success (RMS speed below .01 m/s held
for .5 s), table containment, coverage .15..85 inclusive, the existing .10 edge
extension strain guard, release, and at least one visible feasible grasp pair.
The frozen scene, mesh, material, appearance, runtime/numerical identity is unchanged.
Folds are development-only; they do not establish independent generalization.

Every preparation intent, rejection, exception, and available failed state is
retained. At most 24 drop resets are reserved. A deficient pool stops with
`pool_limitation`; a rejected fold stops with `fixture_limitation`. Such incomplete
manifests remain readable and cannot qualify. `complete` and `stage_passed` are
true only for completed preparation. An uncertain reset is never retried on resume.

`benchmark-manifest.json` and its SHA-256 seal freeze selected states, first-six
order, reset groups/splits, one cloth identity, configurations and content/file
hashes before tuning. State bundles use the existing JSON/NPZ codec, never pickle.
`load_benchmark(output_dir, identity=...)` and `load_fixture(...)` verify the seal,
all referenced files, native state hashes, scene identities and split isolation.
Physical content is hashed independently of seed metadata to reject relabeled
identical states across splits. Arbitrary transformed ancestry cannot be inferred
from geometry alone: callers must preserve frozen reset-group lineage and must
not generate derivatives across splits.

## Budgets and integration API

`BudgetLedger(output_dir).reserve(category, wall_seconds=..., key=..., metadata=...)`
must run before simulator execution. It returns a token for
`finish(token, wall_seconds=elapsed, result=...)`. Reserve the entire operation
wall allowance and enforce it with the existing deadline guard. `status()` returns
counts and remaining wall allowance. `abandon(token, result=...)` reconciles an
uncertain call, charging its full allowance and retaining its consumed attempt.
Unresolved reservations block execution. Keys cannot repeat.

Categories cap diagnostic actions at 26 laydown + 6 fling + 8 continuity (40 total),
pilot actions at 120, drop resets at 24, and cumulative reserved/completed simulator
wall at 21600 s. `simulator` reserves wall for non-action overhead such as restores.
Folds have three reservations. Never count a native multi-action diagnostic as one
action: reserve each attempted action individually in coordinator orchestration.
The ledger is protected against concurrent writes with an OS lock and atomic fsync
replacement. It prevents accidental budget resets across process interruptions;
it is not a security boundary against intentional rewriting of local evidence.

`record_qualification(output_dir, record, identity=...)` freezes the coordinator's
recomputed physical gate and settings. Record fields: `manifest_sha256`,
`benchmark_identity`, `passed` (boolean), `primitive_config` mapping `laydown` and
`fling` to configuration dictionaries, `selection_config`, `laydown_runs`,
`fling_runs`, and `continuity`. Each evidence item has `fixture_id` and
`diagnostic_ref: {path, sha256}`; action evidence also records `primitive_config`,
`pixels`, and `stage` (`initial`/`revision`). References are relative to the benchmark
root. Qualification cannot use heldout fixtures or change during/after pilot.
The coordinator owns physical gate computation and `benchmark_run.py` orchestration.

`claim_pilot(fixture_id, selector, qualification_sha256=...)` irreversibly claims
one episode before restore, including zero-action and restore-failure episodes.
It returns an episode token. Every `pilot` action reservation must have
`metadata={episode_token: token}` and is limited to five per episode.
`finish_pilot(token, result=...)` closes the claim after all simulator reservations
are resolved. Only `uniform_valid` and `geometry_guided` are declared selectors.
All twelve development group/selector episodes must finish before heldout starts.
Each heldout pair can be claimed once, and diagnostics/development execution is
sealed after heldout begins. Geometry-guided selection uses privileged depth;
it is not learned RGB-only behavior evidence.

The fixture agent's CPU tests do not qualify simulator or primitive behavior.
The coordinator runs simulator work serially after integration under its GPU lease.
