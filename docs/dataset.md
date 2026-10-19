# Stage 2 collection and dataset format

This is a local, simulation-only pipeline. Successful verification establishes storage and
episode-chain integrity. It does not qualify reset physics, retained grasp, fling behavior,
contact grasping, or a learned policy. No training or release is included.

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
