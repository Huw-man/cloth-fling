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

## Offline reporting

`report_benchmark(output_dir)` reads pilot datasets under `pilot/**/manifest.json`
and writes `benchmark-report.json` and a self-contained `index.html` with embedded
SVG coverage curves, representative initial/terminal RGB, and relative links to
native videos. It uses no external web libraries and constructs no simulator.
Missing video examples are labeled as unavailable; it never invents capture evidence.
Native grasp spheres are supplied by the existing recorder, not drawn onto RGB by
this reporting module.

Each dataset episode must bind to the frozen benchmark through
`episode.metadata.benchmark` or
`run.metadata.episode_metadata.benchmark` with `manifest_sha256`, `fixture_id`,
`selector`, and `qualification_sha256`. Its metadata also confirms
`initial_settled: true`. The run keeps existing `configuration_provenance` and
`primitive_config`; these must match the frozen scene/appearance/numerical identity
and qualified laydown settings. Optional `selection_config` / `selector_config`
metadata, when present, must match the frozen selection settings. Episode split,
reset-group and cloth IDs must match the fixture. The source snapshot and observed
reset must match the exact paired frozen state, and the first recorded coverage
must match the frozen initial coverage. Dataset hashes and all support files are
verified. Duplicate fixture-selector episodes, changed configurations, altered
states, and changed recording hashes are rejected. Unfinished datasets remain in
the report with explicit incomplete flags and cannot pass the gate.

Declared selectors are `uniform_valid` and `geometry_guided`; the latter's native
transition policy label `geometry_guided_privileged_depth` is accepted and explicitly
reported as privileged-depth-assisted. This is not learned RGB-only policy evidence.
Complete early closures include `settled_coverage_threshold` and
`initially_unfolded`, as well as no-pair/invalid terminal episodes. Interruptions,
timeouts, reset failures and selection failures remain incomplete.

Curves have six points: initial and actions 1..5. A valid terminal updates the
curve only when its evidence explicitly confirms `diagnostics.settled: true`.
Successful early terminal coverage carries forward. An invalid action carries the
last **confirmed settled** coverage (initial if none), while its observed terminal
geometry is separately reported with reason and settling status. Negative valid
gains remain negative. Missing steps and missing episodes carry the frozen initial
or most recent confirmed coverage, retain missing/incomplete flags, and stay in
the group denominator. A per-episode success means final confirmed coverage >= .95;
its success fraction includes missing and failed groups. No-pair fraction uses the
same episode denominator. Invalid/valid attempt rates use actual attempted actions;
missing episodes are reported separately and never become fictitious attempts.

The report compares selectors on each exact frozen reset and gives paired final
coverage differences, both selector curves, final coverage, signed improvement,
success fraction, invalid/no-pair rates, failure evidence and separate recorded /
unrecorded execution timing. Execution row timing excludes reset/restore/selection
and persistence overhead; the durable ledger charges whole simulator operations.

The heldout geometry gate requires six heldout reset groups, all twelve paired
selector episodes complete, mean geometry final improvement >= .10, strictly
positive geometry final improvement in at least four of six groups, geometry valid
attempt fraction >= .80, passed fixture preparation and physical qualification,
and no unresolved integrity or budget evidence. Missing/incomplete episodes cannot
produce a passing gate. The report does not retune settings or rerun heldout.

## Validation delivered by fixture/evidence agent

CPU verification uses synthetic environments and existing DatasetWriter artifacts:
first-six ordering, rejected-fold/pool stops, safe failed/nonfinite state storage,
interruption without replay, immutable document and NPZ hash rejection, split
isolation, durable per-category/six-hour budgets, paired one-shot pilot claims,
negative/invalid/unconfirmed settled curves, missing/incomplete denominators,
privileged selector labels, accepted early closures, frozen config rejection,
offline imports, and both passing/failing heldout gate calculations. These are
software contract checks, not physical behavior qualification. Real simulator
results belong to the coordinator's later serial execution and final results.

A preparation-only report requires no qualification artifact. It includes every
reset attempt and rejection, per-pool accepted/required counts, failed pre-action
checks, and relative links to saved accepted/rejected JSON and NPZ state bundles.
If preparation stops at a pool limitation, qualification/pilot/heldout stages are
**NOT EVALUATED** until corresponding evidence exists. The heldout gate then has
`evaluated: false`, `passed: null`, and `status: not_evaluated`; this is not a failed
behavior result. With no observed pilot episodes, pilot curves are empty and
explicitly missing, final behavior metrics are null, and initial coverage remains
reset-preparation evidence. Missing paired comparisons are null rather than
fabricated zero differences. In mixed observed/missing groups, coverage aggregates
retain missing frozen initial baselines and disclose that accounting in curve status.

Diagnostic references carrying provenance are also checked against frozen scene,
appearance/numerical settings and the fixture state content hash. Provided fixed
pixels and primitive configurations must match their diagnostic source. This does
not replace the coordinator's physical qualification gate computation.
