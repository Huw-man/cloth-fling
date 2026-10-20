# Single-reset RGB grasp overfitting

The user authorized one bounded local campaign to determine whether the current RGB
grasp learner and approved v4 continuation can flatten one exact reset within three
actions. The primary target is normalized projected coverage >=.80, complete tabletop
containment, native release/cleanup, and the existing endpoint settlement criterion:
whole-cloth RMS speed <=.01 m/s for five consecutive controller samples.

The user lowered the primary milestone from .98 to .80 while the first campaign
was running. Its original `single_reset_overfit_v1` protocol and .98 execution
stopping rule remain immutable. Report the first qualifying .80 action within each
saved episode separately, including when later recorded actions lost coverage or
failed settling. This retrospective assessment does not imply that the executed
episode stopped at .80 or that a collected example qualifies the learned policy.
All three frozen greedy/confirmation episodes are still required for qualification.
The amendment grants no additional actions, wall time, or campaign restart.

Future runs use the separately identified `single_reset_overfit_v2_80` protocol and
stop at .80. `--coverage-target .98` preserves the historical protocol for explicit
reproduction in a new authorized output root. Both versions retain the same
immediate-coverage-gain reward, motion, physics and fixed schedule below.

Use archived `fold-50-+00` from the completed image-learning pilot, starting at
.48615 coverage. Copy and hash-verify its complete schema-2 fixture; never regenerate,
substitute or perturb it. Restore at episode boundaries only. The same towel, camera,
physics, native assisted pins and frozen v4 continuation (including 12 cm drag) apply.
This is repeatability on one training reset, not held-out generalization, physical
pinch grasping, literal wrinkle-free geometry or material qualification.

## Frozen schedule

| Stage | Episodes | Maximum actions |
| --- | ---: | ---: |
| Uniform feasible exploration | 20 | 60 |
| Seven refresh rounds, ten episodes each | 70 | 210 |
| One greedy check after each of eight fits | 8 | 24 |
| Two frozen confirmations | 2 | 6 |
| Total | 100 | 300 |

Each episode executes at most three actions and stops when the complete success
condition holds. A failed or unsettled outcome ends its episode without replacement.
Exploration successes do not qualify the learner. The first successful greedy check
freezes the checkpoint and immediately triggers two confirmations. Both must pass;
one failed confirmation ends the experiment without further tuning. Without a passing
greedy check, the six confirmation slots remain unused. No unused slot creates more
episodes, optimizer fits or retries.

Train from scratch with the existing RGB value-map network and settings: seed 1001,
AdamW lr .001 / weight decay .0001, Huber delta .1, batch 16, 2,000 steps per fit.
Require at least 32 eligible settled exploration samples. Refresh actions use 50%
uniform feasible exploration and 50% greedy predictions from the latest checkpoint,
whose weights remain fixed during collection. Continue fitting on the accumulated
exploration/refresh data with fresh optimizer state. Keep signed negative gains;
invalid/unsettled labels remain excluded from regression but their attempts are saved.
Checks and confirmations never enter fitting. Use seed 1001 plus the executed episode
ordinal; the predetermined schedule never replaces failed episodes.

The ceiling is six hours total execution, including preparation, fitting, checking
and reporting. The final hour is reserved for confirmation/evidence. Reserve a full
three-action allowance before beginning an episode. Existing limits remain 35 simulated
and 180 wall seconds per action. Integrity, restoration, ownership, contact-accounting,
rendering, numerical or cleanup regressions stop the campaign. Stable placement quality
is observational, and material strain remains separately reported.

## Execution and evidence

Implement and review in an isolated Herdr worktree; integrate audited paths locally
and hold the exclusive RTX 5080 lease. From the project root:

```sh
UV_CACHE_DIR=/tmp/cloth-fling-uv-cache .venv/bin/python -m cloth_fling.cli learning-overfit run
.venv/bin/python -m cloth_fling.cli learning-overfit report
```

The new root is `runs/single-reset-overfit-v1/`. `run` refuses every existing root
before acquiring a lease or opening a scene. There is no physical-resume command.
`report` is read-only, works without Genesis or Torch, and cannot reopen the run.
Immutable intents precede actions; fixture, source, controller, dataset and checkpoint
identities are recorded. Independent cleanup is saved before reporting.
The default root stays terminal once created; changing the coverage target cannot
reopen it. `report` reads each run's stored execution target and also includes the
primary .80 milestone. Preserve the original executed report when writing the
amended assessment to `primary-milestone-report.json`.

Evidence includes action-wise coverage, predicted and observed gains, failures,
native state continuity, contact accounting and strain. On success, retain the frozen
checkpoint and replays of the greedy check plus both confirmations. Otherwise retain
the best greedy episode (or best collected episode if fitting never began), all raw
failures and the unmet criteria. Replays reconstruct saved geometry without physics.
Return project-relative replay paths without opening viewer panes. Historical campaigns
remain terminal and unchanged.

The first campaign is now terminal and incomplete; see
[`single-reset-overfit-results.md`](single-reset-overfit-results.md). Do not restart
it or use unused budget for additional collection or fitting.
