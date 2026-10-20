# Near-half-fold RGB learning pilot

This separately versioned pilot tests nearby configurations of one near-half-fold
family, using the unchanged user-approved v4 continuation, 12 cm final drag, towel,
camera and contact physics. It learns immediate signed coverage gain from RGB
action outcomes, not expert demonstrations or a learned STOP action.

Every episode ends immediately upon valid, settled coverage >=.80 with complete
tabletop containment, at most three continuous actions. Existing settlement means
whole-cloth RMS speed <=.01 m/s for five controller samples. Restore the exact
schema-2 fixture once at episode entry; never reset between actions.

## Fixed schedule

Prepare twelve centered finite-radius folds with the existing 2 cm crease radius.
Training fractions .46/.50/.54 each use -10/+10 degree orientations. Validation
fraction .48 uses -5/+5 degrees. Test fractions .47/.53 each use -7/+7 degrees.
All fixtures must pass existing settlement, containment, finiteness, deformation,
feasible-action and exact-restoration checks before collecting any action. Freeze
the manifest and splits before learning. No substituted fixtures or rerolls.

| Stage | Episodes | Maximum actions |
| --- | ---: | ---: |
| Random collection: six fixtures, four episodes each | 24 | 72 |
| 50% random / 50% initial-model refresh: same schedule | 24 | 72 |
| Validation: two checkpoints, two fixtures, two repeats | 8 | 24 |
| Test: selected checkpoint and random, four fixtures, two repeats | 16 | 48 |
| Total | 72 | 216 |

Fit from scratch after random collection, then continue weights after refresh using
combined training data and a fresh optimizer. Each fit uses the existing RGB U-Net,
2,000 steps, seed 1001, AdamW lr .001 / weight decay .0001, Huber delta .1 and batch
16. At least 32 eligible settled training samples are required initially. Preserve
negative gains; retain invalid/unsettled source rows with no regression labels.
No historical data/weights, validation or test examples enter fitting.

Validation chooses highest success rate at .80, then mean failure-aware terminal
coverage, then the first checkpoint. Freeze its hash before test. Balance policy
order across the two repeats, use matching episode seeds and exact saved starts,
and retain each repeat independently. Selected-policy 6/8 successes exceeding
random's success count is the provisional signal to consider a larger experiment;
it does not establish broad reliability or authorize further work.

Report cumulative success within one/two/three actions, success count, terminal
coverage, failures, action count among successes, each reset/repeat and material
strain separately. Failed outcomes score zero in the selection metric; retain raw
coverage alongside that score. Repeats measure native variation, not extra independent
reset diversity.

## Execution, limits and evidence

Use an isolated Herdr worktree and exclusive RTX 5080 lease. The campaign ceiling
is six elapsed execution hours and 216 physical action attempts, including failures.
Keep the final two hours unavailable to preparation/collection. Every action retains
35 simulated/180 wall-second limits; reserve a complete action before claiming it.
All failures consume slots. Unsettled/ordinary quality failures end that episode;
existing numerical/deformation, ownership, contact, restoration, artifact, rendering
and cleanup failures stop dependent execution. No motion or physics tuning.

Run with `cloth-fling learning-family run --output runs/near-half-fold-pilot-v1`.
The `run` command refuses every existing root before GPU/backend construction. No
physical resume is exposed. Offline `learning-family report --output ...` audits
intent/row/receipt accounting, including unfinalized episodes, without a simulator.
Source/protocol/checkpoint/fixture/dataset identities are sealed and checked before
dependent stages. Cleanup is written independently of reporting. All historical
campaigns remain terminal.

Deliver verified datasets, the selected checkpoint when available, `family-report.json`,
and saved-geometry overhead/oblique/combined replays of representative outcomes and
available failures. Give project-relative paths only; never open replay panes.
Budget exhaustion or a campaign stop is incomplete, with no replacement attempts.
