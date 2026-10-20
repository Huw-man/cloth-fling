# Near-half-fold pilot: preparation completed, collection blocked

The first bounded run is terminal and incomplete. All twelve fresh fixtures passed
preparation, native settlement, containment, feasible-action and exact-restoration
checks. No fling action was claimed or executed; neither training fit, validation
nor held-out evaluation ran. There is no policy result or action replay.

## Failure and correction

Executed revision `570d1f52137802521c9b6ac2e3dd495287fa13d2` put an outer process
real-time watchdog around every stage. The collector already owns that timer for
reset, selection and action execution. At the first training episode it rejected
the occupied timer before calling fixture restoration:

`RuntimeError: collection requires an unused process real-time timer`

The shared episode wrapper reported the broad reason `fixture restoration failure`.
This was a runner timer-ownership bug, not an observed native-state mismatch or a
cloth physics failure. The failed reset's hashed evidence contains the exact cause.
Its cleanup helpers also rejected the occupied timer; the independent episode and
scene cleanup ran after those contexts unwound and both passed.

The corrected runner leaves collection stages without an enclosing timer. It bounds
fixture/checkpoint preloading separately, keeps collector-owned reset/selection/action
deadlines, reserves complete actions before claiming them, and retains watchdogs
around preparation and fitting. New CPU integration checks exercise real collection
inside all four collection/evaluation stage contexts and verify deadline ownership
and 80% early stopping. Offline reports now expose the original reset error.
No physics, motion, settlement threshold or budget changed. The original run and
its source identity remain immutable.

## Accounting and evidence

- Twelve preparations, all accepted and frozen; fixture hashes independently verified.
- One failed episode initialization; zero action intents and zero transition rows.
- Zero training, validation or test episodes completed; no checkpoint selected.
- Original runner charged 19.873 seconds, including scene cleanup and its report.
- Independent episode and scene cleanup passed; exclusive GPU lease released.
- Failed-reset dataset verification passed. The saved dataset contains the reset
  reference and error evidence, not action-outcome demonstrations.
- No success-rate, coverage-gain or material-behavior claim is available from this run.

Evidence paths from the project root:

- `runs/near-half-fold-pilot-v1/fixtures-manifest.json`
- `runs/near-half-fold-pilot-v1/failure-analysis.json`
- `runs/near-half-fold-pilot-v1/executed-family-report.json`
- `runs/near-half-fold-pilot-v1/corrected-offline-report.json`
- `runs/near-half-fold-pilot-v1/native.log`
- `runs/near-half-fold-pilot-v1/final-verification.json`

The executed `family-report.json`, ledger, fixture bytes and failed episode remain
unchanged. The protocol forbids automatic continuation of failed/uncertain stages;
unused action or time allowances do not reopen this root.

## Corrected execution within the original scope

A one-time transfer into `runs/near-half-fold-pilot-v1-corrected/` carries the exact
prepared fixture bytes and the failed initialization as one consumed exploration
episode slot. It does not repeat preparation or retry that episode. The original
19.873 seconds remain charged against the same six-hour budget. Thus at most 71
new episodes and 213 actions remain; the original 72-episode/216-action ceiling is
not extended. A durable transfer intent prevents reusing the source for another run.
The transfer is allowed only for this proven pre-restoration timer collision with
zero action intents/rows and certified cleanup; physical/uncertain failures are
not recoverable through this path. Results of the corrected execution follow below.
