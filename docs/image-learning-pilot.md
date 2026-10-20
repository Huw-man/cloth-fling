# Bounded image-based grasp learning pilot

This local experiment trains immediate-value RGB grasp selection around the unchanged
user-approved v4 laydown. The approved baseline remains
`configs/end-to-end-pilot-baseline-v1.json`. This is assisted native attachment, not
physical pinch grasping or trajectory learning. Material fidelity and the known
half-timestep contact discrepancy remain limitations, not new placement gates.

## Frozen protocol

The experiment has eight hours of execution time, at most 21 fixture preparations
and at most 432 action attempts. The final two hours are reserved from preparation
and collection for fitting, validation, testing and reporting. No unused budget
authorizes additional cases, parameter searches or retries.

Fold fractions .20/.35/.50 are training families; .30 is validation; .25/.40/.45
are test families. Each has -15/0/+15 degree orientations, all within its parent's
split. Preparation shifts the existing finite-radius construction to the required
long-axis crease with a 2 cm radius, centers the projected footprint and uses the
frozen towel's native material rest geometry, topology and physics. Particle writes
occur only during explicit preparation. Native simulation settles for at most ten
seconds, requiring .5 seconds at RMS speed <=.01 m/s, containment, no pins, finite
state, existing domain/deformation safeguards and a feasible image action. Restore
must reproduce the schema-2 state exactly. All 21 fixtures are frozen before any
action collection. A failed preparation stops without substituting cases.

| Stage | Maximum actions |
| --- | ---: |
| Nine training fixtures, eight random three-action episodes each | 216 |
| Nine training fixtures, five 50%-random/50%-initial-model episodes each | 135 |
| Three validation fixtures, random plus two checkpoints, three actions | 27 |
| Nine test fixtures, random versus selected checkpoint, three actions | 54 |

Each episode restores once, then evolves without reset. Settled coverage >=.95
ends early. An unsettled outcome ends the episode and is preserved. Failed episodes
consume their slot. Ordinary quality failures advance to the next scheduled episode;
numerical/domain/gross-strain, native ownership/contact/cleanup, restoration, source,
rendering and artifact regressions stop dependent execution. Actions retain the
approved 35 simulated/180 wall-second limits. The first exploration episode also
serves as the charged native smoke check.

## Learner

Use the existing 96 rotated/scaled RGB images and feasible action codec. The network
is a shared three-level U-Net with channels 32/64/128, a 256-channel bottleneck,
skip-connected nearest-neighbor decoder, and one unbounded value output per pixel.
Each block has two padded 3x3 convolutions with eight-group normalization and ReLU.
Only uint8 RGB divided by 255 enters the network. Depth remains a host-side
feasibility aid. No pretrained weights, extra augmentation or privileged geometry
channels are used.

Only the executed transform/pixel receives signed settled coverage-change supervision.
Unexecuted pixels are unlabeled. Invalid/unsettled rows retain null learning labels
in this view; their complete attempts stay in the source datasets. RGB reconstruction
uses the exact saved transformation, never a new foreground crop. Dataset verification
and train-only family checks precede fitting; at least 32 eligible exploration
samples are required. Two historical demonstration actions are not training data.

Train with seed 1001, Huber delta .1, AdamW lr .001 / weight decay .0001, batch 16,
and 2,000 steps. Keep the first model fixed during the refresh round. Continue its
weights for another 2,000 steps on the combined data, using freshly initialized
AdamW optimizer state. Validate only the two completed checkpoints. Higher mean
failure-aware terminal coverage wins, with ties selecting the first checkpoint.
Model metadata pins source manifests, weights, preprocessing and frozen motion;
reload must preserve inference exactly. Incompatible/nonfinite predictions stop
before acquisition, with no random fallback.

This follows the value-map / immediate delta-coverage idea from
[FlingBot](https://proceedings.mlr.press/v164/ha22a.html), adapted to this laydown,
not a reproduction of its fling or its results.

## Execution and reporting

From the project root with the locked shared environment and exclusive RTX 5080:

```sh
UV_CACHE_DIR=/tmp/cloth-fling-uv-cache .venv/bin/python -m cloth_fling.cli learning-pilot run
```

Default output is `runs/image-learning-pilot-v1/`. `run` and `prepare` refuse an
existing root. For deliberate staged execution use `prepare`, `collect --stage
exploration`, `train --stage initial`, `collect --stage refresh`, `train --stage
refreshed`, `evaluate --stage validation`, `evaluate --stage test`, then `evaluate
--stage replays`. Only the next unclaimed stage can execute. Its prerequisites and
implementation identities must match. Each stage, episode and action has an exclusive
fsynced intent before native writes. Failed, active or uncertain stages cannot
resume. Offline `report` can inspect them without simulation or Torch, and never
changes their execution ledger or makes them eligible for replay.

The primary metric is paired final settled coverage after at most three actions.
Episodes without a valid settled terminal state score zero for this metric; observed
coverage and reasons are retained separately. Average orientations within a family,
then average the three test families equally. Behavior-positive requires >=.05
mean paired gain, improvement in at least two families, no additional learned-policy
failures, and complete integration/artifact audits. This narrow comparison is not
statistical evidence of broad reset generalization. Pipeline completion is reported
separately from behavior-positive status.

Every episode saves native snapshots, commands, contact accounting, geometry,
cleanup, image/action metadata and signed rewards through the existing durable
collector. Independent cleanup is written before reporting. Reports include strain,
settlement and containment diagnostics without hiding negative coverage gains.
Save overhead/oblique/combined replays for the lexicographically first test family
under both policies, plus every other failed test episode. Their paths are relative
to the project root; no viewer panes are opened. Historical terminal campaigns,
approved controller hashes, physics and contact settings remain unchanged.
