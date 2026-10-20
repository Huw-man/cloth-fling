# Image-based grasp learning pilot results

The bounded pilot completed collection, two model fits, validation selection and
the held-out comparison with the approved v4 laydown and unchanged physics.
The first checkpoint met the predeclared **relative improvement** criterion on
the test set. Mean terminal coverage was .476394 versus .351837 for uniform
feasible grasps, a gain of .124557. All three test families improved, with no
failed test episodes under either policy.

Mean initial test coverage was .622646. Both policies reduced coverage overall;
reliable unfolding remains unmet. This is a small simulation-only ranking result
from three held-out fold families, not broad policy, material or physical pinch
grasp qualification. Validation results and material limitations below remain
part of the evidence.

## Protocol and accounting

The experiment is terminal at `runs/image-learning-pilot-v1/`. It prepared exactly
21 fresh schema-2 fixtures: seven fold families with three orientations each.
Family-isolated splits, fixture hashes and exact restoration checks were frozen
before collection. The test contains nine paired starts, grouped into three
families. States evolve without reset within each episode.

| Stage | Episodes | Actual actions | Failed episodes |
| --- | ---: | ---: | ---: |
| Uniform training exploration | 72 | 190 | 26 |
| 50% random / 50% first-checkpoint refresh | 45 | 118 | 10 |
| Validation: uniform and two checkpoints | 9 | 23 | 1 |
| Test: uniform and selected checkpoint | 18 | 52 | 0 |
| Total | 144 | 383 | 37 |

All 383 physical actions returned `completed` with valid execution. The 37
failed episodes failed settlement; they remain recorded and receive zero in
failure-aware episode evaluation. Their actions are not silently relabeled as
valid settled training targets. There were 249 negative action coverage gains.
No failed episode was replaced, and no action was retried.

Active preparation, collection, fitting, evaluation and replay phases charged
15,815.579 seconds (4.393 hours), below the eight-hour ceiling. The final offline
artifact check brought elapsed campaign time to 4.516 hours; final independent
verification finished at 4.547 hours, also below the ceiling. The
maximum action used 35.000000000146 simulated seconds (floating-point roundoff
at the 35-second limit) and 47.691 wall seconds, below the 180-second wall guard.
No unused budget was used for tuning or extra collection.

## Learning and selection

The initial fit used 164 valid settled labels. The refreshed fit used 272 labels
from the two training rounds, with no validation/test data entering fitting.
Both ran the frozen 2,000-step, seed-1001 configuration and passed exact inference
reload checks. Training Huber loss decreased .018820 → .000185 for the first fit
and .012994 → .000243 for the refreshed fit. These are fitting diagnostics.

| Policy | Mean validation terminal coverage |
| --- | ---: |
| Uniform feasible | .625028 |
| Initial checkpoint | .394806 |
| Refreshed checkpoint | .246750 |

Validation selected the initial checkpoint between the two learned candidates;
uniform selection performed better on that validation family. No model or
trajectory was tuned after seeing these results. The selected weights are
`runs/image-learning-pilot-v1/checkpoints/initial.pt`, SHA-256
`6f1819a392ee01e0152871d4be8fa55144f98dd9c57e4343e7c9e3687ff65579`.
The refreshed checkpoint and its negative validation evidence are preserved.

## Held-out behavior

| Fold family | Mean paired learned-minus-uniform terminal coverage |
| --- | ---: |
| .25 | +.040200 |
| .40 | +.083294 |
| .45 | +.250178 |
| Equal-family mean | +.124557 |

The learner improved seven of nine individual pairs. It reached the .95 settled
coverage stop in one of nine test episodes, versus zero for uniform selection.
The successful learned case ended at .982783 after one action. These few cases
do not establish statistical generalization; the opposite validation ranking
also limits confidence. The principal remaining behavior problem is the decline
from starting coverage even under the selected learner.

Material fidelity remains unqualified. Peak local edge-length ratios were
2.826306 during exploration, 2.970057 during refresh, 2.404369 in validation and
2.918348 in test. Report these separately from projected coverage and numerical
stability. The known half-timestep table-friction discrepancy remains unresolved;
the pilot preserved the approved normal-timestep physics.

## Verification and artifacts

The native implementation was local commit `347c640`. Its package source hashes,
runtime, immutable motion specification and preflight checks are archived in the
run. Subsequent deadline/accounting guards are offline corrections and do not
change or rerun this native experiment.

Independent accounting reconciled all 383 durable action intents, ledger entries
and dataset rows. All 383 independently written action cleanup certificates and
five scene cleanup certificates passed. Peak pin tracking error was 4.20e-8 m.
Native release velocity preservation, state-chain equality, saved-command drag
continuity and dissipative contact accounting passed the runtime audits.

The final implementation passed 546 CPU tests (eight GPU tests deselected), plus
14 focused deadline/accounting/lifecycle checks. All 18 overhead/oblique/combined
videos decoded fully; representative saved frames were visually inspected without
opening a viewer pane. The GPU lease was released after native work.

Project-root-relative entry points:

- Report: `runs/image-learning-pilot-v1/pilot-report.json`; its preserved executed
  copy is `runs/image-learning-pilot-v1/executed-pilot-report.json`, SHA-256
  `0ba972e35ef667cdcf1cfff2d0bff51e57818fae802c081411a1a9cb2c6a77be`.
- Protocol, fixture manifest and durable budget: `protocol.json`,
  `fixtures-manifest.json` and `ledger.json` within that run.
- Verification: `independent-accounting-verification.json`,
  `video-verification.json` and `verification.json` within that run.
- Selected learned replay:
  `runs/image-learning-pilot-v1/replays/test-fold-25--15-initial/combined.mp4`.
- Its exact-start random comparison:
  `runs/image-learning-pilot-v1/replays/test-fold-25--15-uniform/combined.mp4`.

The replay family was selected by its identifier before reviewing test outcomes;
it is representative evidence, not a best-case montage. All three paired
orientations have separate overhead, oblique and combined files. No test episode
failed, so no additional failed-test replays were needed. Further training,
collection or changes to the motion require a new bounded experiment.
