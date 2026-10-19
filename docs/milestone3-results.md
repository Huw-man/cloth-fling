# Milestone 3 preparation limitation — 2026-10-19

The bounded benchmark stopped at its first gate. Fresh Genesis 1.4.4/native-v2
preparation supplied six development drop bunches but only five held-out bunches
from the twelve permitted held-out candidate seeds. The user required stopping
if either pool supplied fewer than six. No motion tuning, laydown/fling diagnostic,
continuity episode, pilot collection, learning, publication, cloud or hardware
execution followed. Held-out behavior is **not evaluated**, with no pass/fail
claim about laydown or selection quality.

## Frozen inputs and observed reset outcomes

The scene retained the migration's 0.50 m square cloth, 1 cm mesh, native-v2 contact,
material, matte appearance, workspace and numerical settings. Physical states use
schema 2 and explicit Genesis 1.4.4/runtime identities. The manifest freezes
candidate order, accepted and rejected state references/hashes, checks, split,
reset-group and cloth identity. Every available failed snapshot is retained.

| Pool | Attempts | Accepted | Required |
| --- | ---: | ---: | ---: |
| Development folds 11/17/23 | 3 | 3 | 3 |
| Development drops | 8 | 6 | 6 |
| Held-out drops | 12 | 5 | 6 |

Development drops accepted the first qualifying seeds **1000, 1001, 1003, 1005,
1006, 1007**. Seeds 1002 and 1004 exceeded 85% initial coverage. The development
pool stopped at six; seeds 1008–1011 were not attempted.

Held-out drops accepted **2000, 2001, 2002, 2004, 2005**. All permitted candidates
were exhausted:

| Rejected held-out seed | Measured rejection |
| --- | --- |
| 2003 | Existing bounded settling budget exhausted |
| 2006 | Initial coverage 37.6625%; peak edge strain 10.191214%, above the unchanged 10% guard |
| 2007 | Initial coverage 99.9975% |
| 2008 | Initial coverage 90.9275% |
| 2009 | Initial coverage 99.9975% |
| 2010 | Initial coverage 100.0450% |
| 2011 | Initial coverage 100.0000% |

The coverage estimator is retained unchanged, including its small rasterized
overshoot above 100%. Such states clearly exceed the 85% preparation ceiling.
No damping, physics, material, acceptance threshold or generator parameter changed
to improve acceptance. No rejected state was substituted or relabeled.

The existing suspension/drop generator frequently produces cloth that is already
nearly flat. Under this frozen candidate pool and unchanged acceptance criteria,
it does not provide the required held-out distribution. This is a reset-generation
limitation, preceding any evidence about motion quality or generalization. A later
generation protocol would need its own reviewed design and fresh benchmark; this
manifest is not extended with extra seeds or a relaxed guard.

## Implementation and evidence boundary

Three requested Codex agents worked in separate Herdr worktrees. Their reviewed
fixture/budget/report, primitive diagnostic, and collection/data commits were
integrated into local main. Coordinator integration adds the `benchmark prepare`,
`qualify`, `pilot`, and `report` command stages, exclusive simulator lease, gate
checks and interruption/resume accounting. The transition schema remains version 1;
benchmark identity and settlement bookkeeping use episode metadata/support files.
Genesis/Torch imports remain lazy for offline work.

The integrated CPU suite passed **420 tests**, with eight GPU tests deselected.
Checks cover dataset lineage, explicit metadata, settled-only stopping and
aggregation, invalid terminal geometry, signed negative gains, configuration/state
hash rejection, paired accounting, interruption retention and fail-closed resume.
These checks do not qualify physical motion, continuity or policy behavior.

The real run used **20 of 24 permitted drop attempts**, three development folds,
**zero diagnostic actions**, and **zero pilot actions**. The ledger charges
251.512 s for scene construction and bounded reset work. This timing is preparation,
not collection throughput; artifact writes and offline report generation are
separate. No action recordings, transition Parquet or coverage-versus-action
behavior curves exist because their prerequisite gate did not pass. Offline saved
geometry plots are explicitly labeled and are not native RGB or new execution.
The RTX 5080 lease was released and its compute-process list was empty afterward.

All local evidence is ignored under `runs/milestone3/`:

- `benchmark-manifest.json` and `.sha256`: immutable inputs, rejection history,
  runtime/configuration identity and state hashes.
- `fixtures/`: safe JSON/NPZ bundles for all 23 preparation attempts.
- `benchmark-budget.json`: durable reservations, consumed attempts and timings.
- `benchmark-report.json` and `index.html`: incomplete-run report, every preparation
  result, state links and explicitly unevaluated behavior metrics.
- `analysis/`: reproducible offline acceptance and saved-geometry plots.
- `offline-stop-verification.json`: prepare/qualify/pilot return the frozen stop
  without Genesis/Torch import or changes to the manifest, ledger or lease.
- `prepare.log`, `final-cpu-tests.log`, `summary.json`: execution log, integrated CPU
  checks and final artifact inventory.

Re-running `cloth-fling benchmark prepare`, `qualify`, or `pilot` against this
manifest returns `pool_limitation` offline. `benchmark report` can regenerate the
review without simulation. The milestone remains incomplete at preparation.
