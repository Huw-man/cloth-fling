# Collection and dataset format

This is a local, simulation-only pipeline. Successful verification establishes storage and
episode-chain integrity. It does not qualify reset physics, retained grasp, fling behavior,
contact grasping, or a learned policy. No training or release is included.

## Genesis 1.4.4 compatibility boundary

Package 0.2.0 upgrades the simulator; it does not rewrite historical datasets. Parquet,
PNG and safe NPZ/JSON support files remain readable with the base install, without
Genesis, Torch or video encoders. `dataset verify` and `replay --inspect` check saved
artifacts without simulating. Their success is storage/inspection evidence only.

Physical snapshots move to schema 2 with engine identity. Reading an old state is
permitted; restoring it under 1.4.4 is not. Saved-reset collection must pass the backend's
schema, engine, topology and physics-configuration validation before physical writes.
Do not relabel a schema-1 snapshot or settle it silently to make it compatible. An explicit
future geometry derivation would need parent identity and fresh qualification.

New evidence records installed engine/dependency identity separately from source snapshot
identity, the contact profile, rendering backend and backend-reported capabilities.
Unknown capabilities remain unknown rather than inheriting a historical pass. Replay of
historical geometry requires an explicit rendering-only path and compatible topology;
original images and source engine metadata stay distinct from newly rendered frames.

Collection and primitive campaigns remain paused during migration. The interfaces below
document storage and lifecycle behavior, not permission to resume. All Genesis 0.4.3
contact, rendering and experimental laydown outcomes are historical. See the
[migration record](genesis144-migration.md) and [archived results](milestone2-results.md).

## Collection API

```python
from cloth_fling.collection import collect_episodes

report = collect_episodes(
    env, output_dir,
    episodes=1, max_actions=3, seed=0,
    policy="uniform_valid", primitive="laydown", primitive_config=None,
    selector=None, executor=None, max_wall_seconds=600,
    reset_state=None, reset_kind="drop",
)
```

`env` implements `cloth_fling.contracts.ClothBackend`. Default dependencies are imported
only when collection is called:

```python
selection.select_action(observation, env.config, policy=policy, primitive=primitive, rng=rng)
sim.primitives.execute_action(env, action, config=primitive_config)
```

Tests can inject callables with those exact signatures. Selectors return `GraspAction`,
or `None` when no pair is available. A selector `ValueError` is also recorded as
`no_valid_pair`; other selector errors become `selection_error`. Selection does not
constitute an executed action, so these cases close episode metadata without inventing
a transition. Even a zero-action episode retains its reset/configuration evidence.

At each episode boundary collection calls `env.reset(seed=episode_seed, kind=reset_kind)`.
If `reset_state` is provided, it instead calls `env.restore(copy_of_reset_state)`.
The saved reset input and the observed post-restore snapshot are both retained. Generated
reset groups derive from generator kind, seed, cloth identity and configuration hash,
so numerical snapshot perturbations do not change the group. Saved-reset groups derive
from the exact saved input plus cloth and scene configuration; different
policy seeds on the same saved reset remain in the same group. Within a run, and after
reopening the output, `episode_seed = seed + global_episode_id`.

There is no reset, restore, extra observation, or physics step between valid actions.
The next selector receives the exact previous `outcome.after` object. The collector uses
that same object as the next transition's authoritative `before`, even if an executor
constructs a fresh pre-observation internally. Storage reuses the previous RGB and support
references, rather than saving a second pre-image.

The library permits `0 <= max_actions <= 10` per episode. The total wall budget is shared
by reset, selection, execution, and all episodes; the remaining total is also each call's
deadline. The collector requires a POSIX main thread with an unused real-time signal timer.
Signal deadlines interrupt Python work; a native GPU kernel may return before Python can
deliver the signal. Durable filesystem commits finish outside the signal deadline so an
already attempted action is not lost. Simulation-only failure cleanup has at most one
additional second for `release()`, observation, and snapshot. Total elapsed time, including
cleanup and writes, is reported; this is not a hard kernel or filesystem latency guarantee.

For an execution exception or timeout, collection attempts assisted `env.release()` and
captures a terminal observation/state without stepping, resetting, or retrying execution.
Cleanup failures are stored as evidence. Invalid executions end the episode after their
row is committed. A timeout truncates the episode and ends collection; an ordinary invalid
execution permits the next requested episode to start. A failed reset/restore or initial
snapshot creates a distinct failed, zero-step episode and stops the report explicitly,
preserving previously completed records. Failed reset preparation is never an action row.

Episode endings distinguish `max_actions`, `max_wall_seconds`, `selection_timeout`,
`no_valid_pair`, `selection_error`, `execution_invalid`, `environment_terminated`,
`reset_failed`, `reset_timeout`, and recovered `interrupted` work. A boundary discovered
between actions closes episode metadata; immutable earlier rows remain unchanged.

The report contains integer `attempts`, `valid`, `invalid`, `episodes`, `reset_failures`,
the current `run_id`, `stop_reason`, `episode_reports`, and `timings` with total wall,
reset, selection, and execution seconds. These are pipeline counts, not behavior scores.
Recovered old intents appear in the dataset, not in the new run's attempt counts. The
manifest records Python/distribution versions, collection parameters, and run lineage.
Callers using `DatasetWriter` directly may add a code revision/runtime qualification in
metadata; absence of such metadata does not imply runtime qualification.

## Writer API

```python
from cloth_fling.data import DatasetWriter

with DatasetWriter(output_dir, metadata={"split": "train", "code_revision": "..."}) as writer:
    episode_id = writer.write_episode_reset(
        initial_observation,
        seed=0, config=env.config, state=initial_snapshot,
        split="train", reset_group_id="family-a", cloth_id="cloth-a",
        # Optional reset_id/config_id override their content-derived names.
        reset_source=None, reset_kind="drop",  # Saved restore input, if any, and generator kind.
    )
    writer.begin_attempt(episode_id, action, before=initial_observation)
    outcome = executor(env, action, config=None)
    row = writer.append_outcome(episode_id, outcome, final_state=env.snapshot())
    # If the outcome did not end the episode, close its metadata explicitly:
    writer.finish_episode(episode_id, reason="max_actions", truncated=True,
                          final_state=env.snapshot())
```

`save_reset` aliases `write_episode_reset`. `config` may be a dataclass or a JSON-compatible
mapping. `state`, calibration, diagnostics, and evidence may contain nested NumPy arrays.
Arrays are saved without pickle in NPZ; nested structure lives in JSON. JSON nonfinite
scalars are explicit `{"__nonfinite__": "nan"}` / `inf` / `-inf` markers. Object arrays and
unsupported Python objects are rejected.

`write_episode_reset` also accepts optional `reset_id`, `config_id`, and `reset_source`.
Its default split is metadata's `split`, or `train`; default group is metadata's
`reset_group_id`, or a hash of generator kind/seed/cloth/config for generated resets,
or saved input/cloth/config for restored resets. `reset_id` defaults
to the exact state/config content identity and `config_id` to the canonical configuration hash.
`cloth_id` defaults to metadata's `cloth_id`, or `cloth-default`; heterogeneous cloth
collections must supply meaningful IDs. Caller-supplied group labels do not bypass the
verifier's generator family, saved input or exact reset identity/content leakage checks.
Split assignment is recorded,
not randomly changed by this library. No training-ready split distribution is implied.

`begin_attempt` durably records the action and exact pre-reference before execution.
`append_outcome` accepts an `ActionOutcome`, or `None` for a failed journaled attempt:

```python
writer.append_outcome(
    episode_id, None, after=terminal_observation_or_none,
    reason="execution_exception: ...", final_state=terminal_state_or_none,
    terminated=True, truncated=False, end_reason="execution_invalid",
    failure_evidence={"release_error": None},
)
```

`terminated`, `truncated`, and `end_reason` may also override flags for a regular outcome,
for example truncation at the collector limit. `append_outcome` returns the committed row.
Direct offline conversion can append an outcome without `begin_attempt`; execution-based
callers must journal first. No known action is retried automatically. A missing after
observation is null and is only allowed as an invalid terminal outcome; no pre-image is
relabeled as a post-image. Invalid rows always have null reward. Valid negative coverage
gains remain negative. Nonfinite/missing valid metrics downgrade the outcome to invalid;
nonfinite scalars in the Parquet metrics become null and raw values remain in support.

For failed preparation use `write_reset_failure(seed=..., config=..., reason=...,
state=None, reset_source=None, reset_kind="drop", evidence=None)`. It returns a new episode ID with no
initial observation and zero action steps. Use `finish_episode(episode_id, reason=...,
terminated=False, truncated=False, final_state=None, evidence=None)` for boundaries
discovered before the next action; supply at least one terminal flag.

The writer is a context manager; `close(interrupted=False)` is idempotent. It holds a
local advisory single-writer lock. A failed write poisons that instance: close and reopen
to reconcile disk state, rather than attempting the same action again.

## Schema v1 and files

`transition_schema()` is the explicit PyArrow schema, also embedded in each Parquet part
with `cloth_fling.schema_version=1`. Every part has exactly one attempted high-level
action. Numeric metrics are float64; IDs/steps/seeds are int64, version int32, flags bool,
and references/labels strings. Nullable values represent unavailable data.

| Columns | Meaning |
| --- | --- |
| `schema_version`, `transition_id` | Version and global sequential attempt index |
| `run_id`, `parent_run_id`, `episode_id`, `parent_episode_id`, `step_id` | Run and episode ancestry; zero-based sequential episode and local step IDs |
| `split`, `reset_group_id`, `cloth_id`, `reset_id`, `reset_sha256`, `config_id`, `config_sha256`, `seed` | Reset/config identity, artifact hash, group/split, RNG seed |
| `before_observation_id`, `after_observation_id` | Episode-scoped observation identity; unknown after is nullable |
| `{before,after}_{rgb,support}_{path,sha256}` | Independent RGB PNG and privileged observation JSON references/hashes |
| `action_u0`, `action_v0`, `action_u1`, `action_v1`, `primitive`, `policy` | Two image pixels: top-left origin, u right, v down |
| `area_before_m2`, `area_after_m2`, `area_gain_m2`, `flat_area_m2` | Unclamped initial/final projected areas, signed final-minus-initial gain, flat reference |
| `coverage_before`, `coverage_after`, `coverage_gain`, `reward` | Area / flat area and signed gain; reward equals gain only on valid executions |
| `execution_valid`, `reason`, `terminated`, `truncated`, `end_reason` | Execution validity and episode boundary cause, kept distinct |
| `simulated_seconds`, `wall_seconds` | Executor timing; null if unavailable after failure/interruption |
| `final_state_path`, `final_state_sha256`, `evidence_path`, `evidence_sha256` | Privileged state and diagnostics/raw outcome evidence |

The root contains `manifest.json`, `DATASET_CARD.md`, `transitions/part-<unique>.parquet`,
`rgb/<unique>.png`, `support/<kind>-<unique>.json`, and optional matching NPZ. The manifest
contains file checksums/sizes, ordered parts, observations, runs, full episode metadata,
and optional in-flight intent. Reset snapshots, seeds, full configuration, restored input,
and final episode evidence live in the manifest-referenced support files. Every row's
reset hash references its episode's reset JSON; its config hash is the canonical config
content hash. Support JSON references its NPZ with a separate file checksum.

All interpreted artifact paths are relative to the dataset root. Absolute paths,
parent traversal, backslashes, and symlinks escaping the root are rejected. IDs never
become path components. Arbitrary executor evidence is stored as payload, not followed
as an external file path; callers should pass arrays/JSON content, not external filenames
as a substitute for captured evidence. Spatial units are world metres with z up.

Only RGB PNG files are learner-facing input. Depth, calibration, positions, velocities,
reset/config/support state, attachments, metrics and diagnostics are privileged execution
or evaluation artifacts, never an RGB policy observation.

## Atomic commits and recovery

The local filesystem is the only storage backend. Artifacts have unique immutable names;
each append writes/fsyncs support and a Parquet part, then atomically replaces/fsyncs the
manifest. The manifest replacement is the row's commit point. Readers use only registered
parts in manifest order, never a glob. Earlier committed records survive interruption.
The manifest itself is not cryptographically signed; checksums detect artifact corruption
and verification checks internal consistency, not adversarial rewriting of all evidence.

If interruption precedes manifest commit, leftover files are orphans and do not become
rows on reopen. Verification lists them without treating them as committed data. If an
intent was durable, reopening records exactly one invalid `interrupted_in_flight` row
with null reward and unknown post-state, closes its episode, and preserves the old run
identity. It never replays execution. If interrupted between actions, reopening truncates
episode metadata without manufacturing a new attempt. A new writer starts a fresh run
with parent-run lineage; new collection always starts a fresh episode ID. Neither action
nor simulation continuation is automatically resumed. Reopen also checks hashes of all
registered files before extending the dataset. Disk corruption is not repaired silently.

## Offline reading and verification

```python
from cloth_fling.data import read_transitions, read_rgb, read_support, verify_dataset

result = verify_dataset(output_dir)
assert result["valid"], result["errors"]
table = read_transitions(output_dir)  # PyArrow Table, including invalid attempts.
valid_table = read_transitions(output_dir, valid_only=True)
rgb = read_rgb(output_dir, table["before_rgb_path"][0].as_py())
```

`load_transitions` aliases `read_transitions`. RGB loading needs neither the manifest nor
support files. `read_support(root, relative_json_path)` reconstructs safe NPZ arrays.
These modules, including importing `collection`, never import Genesis, torch, selection,
or primitives at import time. Readers do not implicitly verify every file; call
`verify_dataset` before relying on a dataset.

Verification returns `valid` and `ok` booleans, `errors`, `attempts`, `valid_attempts`,
`invalid_attempts`, `episodes`, and (when readable) `orphan_files`. It checks schema,
file existence/sizes/checksums, referenced-file registration, safe paths and NPZ loading,
coverage/gain/reward arithmetic, finite/null handling, sequential global/episode/step IDs,
run ancestry, exact pre/next image/support continuity, initial/final observation identity,
terminal boundaries, reset/config hashes and seed consistency, and reset/group split
leakage including zero-step episodes, numerically varying generated states from the same
seed/family, and saved inputs reused under renamed groups. When supporting final state contains `attached_ids`,
`attached_particle_ids`, `attachments`, or `attachment_count`, attachments must be zero.
When the backend uses a `free` mask, every entry must be true. Supporting final states
for valid executions must include finite `positions` and `velocities`; invalid execution
states may retain nonfinite arrays as failure evidence.
A corrupt/missing dataset returns `valid=False` and an error, suitable for a nonzero CLI exit.
An unresolved intent or open episode/run fails verification until closed or recovered.

Each new dataset includes an immutable dataset-card template with owner/version,
distribution, split, qualification, failure/bias, intended-use, and license fields.
Completing or publishing a dataset card and training a policy are later work.

## Milestone 2 evidence and recording

`collect_episodes` also accepts `record_video=False`, `appearance=None`, and
`recorder_factory=None`. Without recording, injected executors still receive exactly
`executor(env, action, config=primitive_config)`. Recording adds the keyword
`trace_callback`, called synchronously as `callback(env, trace)`; it must not step or
mutate physics. No video dependency is imported when recording is disabled.
`appearance` records the settings used to construct `env`; the collection API does
not change an existing scene's material. CLI simulation commands construct matte
appearance by default, while `--legacy-appearance` reproduces the old appearance.
`--appearance-config` accepts independent render-only JSON. SceneConfig physics and
its hash are unaffected by appearance selection. Direct library/backend callers
retain legacy appearance with `None`.

Historical milestone-2 recording used `configs/contact-native-v1.json` and saved
Genesis 0.4.3 resets. Those are archival inputs; do not execute the old diagnostic or
collection commands with the 1.4.4 environment. New native-v2 resets and all physical
acceptance gates must be established independently before collection resumes.
CLI `diagnose` returns nonzero for invalid execution; `collect` returns nonzero for
invalid actions or reset failures. Neither a zero exit nor a valid dataset qualifies
behavior. `configs/primitive-legacy.json` disables clearance for historical comparison;
`configs/primitive-default.json` retains the 20 mm / 100 ms clearance gate, 50 mm
increments and 750 mm cap, all requiring new-runtime validation.

Recording uses native execution captures labeled `native_render_capture`. Each actual
global episode/action ID starts a separate recorder under
`video/episode-<episode_id>-action-<step_id>/`; continuing a dataset uses new IDs.
Diagnostics use `video/diagnostic-laydown/` and `video/diagnostic-fling/` and require
fresh recording directories. Each directory contains overhead, oblique and combined
MP4s plus metadata and a preview. Capture/encoder failures are explicit invalid
attempts: the collector closes the recorder once, releases, records terminal evidence,
and ends that episode. It never calls the physical action again because closing a
file failed. Partial files are preserved. Missing optional encoders fail explicitly.
Video `complete` means the capture lifecycle completed, independently of motion validity.
A fully recorded `lift_max_height` or strain failure retains its original reason;
`action_execution_valid` and `action_reason` accompany separate recording status.
An encoder failure invalidates an otherwise valid outcome, and preserves an already
invalid primitive's original reason in the action row and supporting recording evidence.
Debug captures remain privileged artifacts; original policy RGB/depth stay clean.

Videos are currently **outside the dataset manifest**. Row evidence and the persisted
`collection-report-<run_id>.json` include relative filenames and SHA-256 hashes under
`recording`/`recordings`; the manifest does not register the MP4s. External references
use `relative_file`, avoiding the writer/verifier's reserved `path` + `sha256` pair.
`dataset verify` checks dataset integrity, not these external files; compare their
reported hashes separately. Do not move external video metadata into a structured
`path` + `sha256` reference unless it is registered through the writer.

Configuration provenance stores physics, appearance and primitive settings separately,
with independent physics/appearance hashes. Source reset content hashes, runtime source
revision/dirty status, lock hash, seed and detected GPU accompany run evidence. Dataset
reset metadata remains the authoritative per-episode reset identity. Genesis contact
accounting in primitive traces is cumulative within its reported epoch; recording does
not implicitly reset it. Local outputs should live under ignored `runs/`.

## Contact qualification and replay commands

Offline inspection remains available during migration:

```bash
uv run --locked cloth-fling replay runs/old-dataset --inspect
uv run --locked cloth-fling dataset verify runs/old-dataset
```

The preferred upgraded contact configuration is `configs/contact-native-v2.json`.
Contact probes require integrated runtime code and the coordinator's exclusive GPU lease;
see [runtime checks](runtime.md). Historical rendered replay requires an explicit
recorded-geometry-only path from the rendering integration, rather than restoring old
physical snapshots. CLI options and capability limits are in [viewer docs](viewer.md).

`diagnose-contact` accepts separate physics JSON, `--duration-s` in [0.3, 2] (default
0.6), and performs the half-timestep comparison by default. A failed physical gate
returns nonzero and preserves `report.json` and contact artifacts. `--no-half-dt` is
exploratory only and always returns nonzero with `qualification_passed=false`.
Unresolved native normal-energy injection blocks primitive qualification. Static and
kinetic priors 0.40/0.30, and sensitivity profiles, are not measured terry/desk values.
This command uses the contact agent's scene-local diagnostic; it does not run primitives.

Replay accepts an existing dataset root or diagnostic JSON/directory, including legacy
saved trajectories. `--episode-id` selects one dataset episode; `--fps` and
`--max-frames` bound rendered output. `--inspect` validates saved artifacts offline
without Genesis, Torch or an encoder. Rendering writes a fresh directory outside the
source and labels output `reconstructed_replay`; it never reexecutes physics. Sparse
saved meshes use the last sample at or before each frame time, without interpolation;
missing trajectories are explicitly terminal-only views. `replay-provenance.json`
records source/video hashes, original-image copies, appearance and runtime identity.
Replay is inspectability evidence, not a new physical execution or behavior test.

Across changed contact/physics configurations, an old snapshot is **not** a valid exact
restore fixture. For the coordinator's bounded comparison, derive geometry in the new
profile, record the parent reset hash, settle once and validate that derived fixture;
then restore it exactly within that profile for every comparison. Settling under the
new profile does not establish identical initial states across profiles. This stage
adds no cross-profile reset import command.


## Experimental full-lift laydown profile

`configs/primitive-full-lift-laydown-v2.json` is a complete `PrimitiveConfig` document
for bounded experimental laydown checks. It matches the resolved
`configs/primitive-default.json` settings except `laydown_distance_m=0.30` instead of
0.12. Drag remains 0.06 m; swing, pullback and placement distances retain their reference
values. The profile keeps all clearance, strain, speed, acceleration, settling and action
guards unchanged. It does not configure physics or appearance. The dataclass/reference
default remains 0.12 m, and `configs/primitive-legacy.json` remains explicit milestone 1
reproduction with clearance disabled.

The coordinator's historical **Genesis 0.4.3** native-v1 checks with full clearance reported:

| Fixed fixture | Final coverage | Interpretation |
| --- | --- | --- |
| Seed 11 fixture, reviewed pair | Approximately 0.99995 | Bounded laydown pass, low strain |
| Frozen seed 17 fixture, reviewed pair | Approximately 0.9999 | Bounded laydown pass, low strain |
| Seed 23 fixture, old grasp pair | Initial lift fails | No completed laydown pass |

These historical observations cover two verified fixed cases; the 0.30 m profile is
unqualified on Genesis 1.4.4. They do not establish a universally
successful profile, a qualified grasp selector, a reset distribution, repeated-action
success or broad-collection readiness. Dynamic fling remains unqualified. A freshly
generated reset with the same seed and an automatically selected pair is not the frozen
fixture/pair comparison. Retain failed attempts and use exact restoration within each
physics profile. The coordinator owns the final receipts and qualification report;
these interim observations do not establish that all milestone 2 gates have passed.
