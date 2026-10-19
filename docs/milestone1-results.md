# First milestone: implemented and locally validated

2026-10-19. Code is integrated on local `main`. No repository/artifact publication,
training, paid compute, or hardware execution was performed.

## Delivered

Three Codex agents implemented stages in separate Herdr worktrees, then the coordinator
reviewed, integrated and validated their work:

| Stage | Agent / branch | Delivered |
| --- | --- | --- |
| 0 | `cloth-stage0` / `codex/stage0-genesis` | Genesis scene, visible-surface attachments, mesh/coverage, reset/restore, exact rendering, GPU checks |
| 1 | `cloth-stage1` / `codex/stage1-primitives` | Controlled laydown, retained-grasp fling candidate, conservative selectors, matched-fixture diagnostic |
| 2 | `cloth-stage2` / `codex/stage2-dataset` | Evolving-cloth collection, Parquet/PNG/NPZ support, verification, interruption recovery |

The coordinator supplied package/CLI/locks, contact-physics adaptation, finite-radius
reset correction, cross-stage integration and host validation. Agent branches are
retained. Git has no configured remote; merges are local, with no push.

## Actual episode and primitive evidence

Host: NVIDIA RTX 5080, 16 GB, driver 595.91.07. Python 3.12.13, Genesis 0.4.3,
Torch 2.11.0+cu128, Quadrants 0.4.4. Exact dependency artifacts are pinned in `uv.lock`.
Use the declared 50 cm synthetic cloth, 64 stretch sweeps, explicit table friction,
2 ms outer dt/four substeps, zero air drag, and 20 mm near-table placement.

Local working artifacts are ignored by Git:

- `runs/milestone1/reset-final/`: saved, settled fold seed 11, state/config/camera and RGB.
- `runs/milestone1/episodes-qualified/`: one episode, two consecutive high-level actions.
- `runs/milestone1/primitive-comparison/`: fixed-pair baseline/candidate comparison,
  before/after images, depths, fixture snapshot and detailed diagnostic traces.
- `runs/milestone1/summary.json`: compact measurements and artifact references.
- `runs/milestone1/convergence.json`: bounded numerical/preparation diagnostics.

The collector restored the saved state once, then selected again on the evolving cloth
without resetting. Both actions released and settled. The exact after-observation of
step 0 is step 1's before-observation, including identical stored image references.

| Action | Valid | Coverage before | Coverage after | Peak edge strain | Execution wall time |
| --- | --- | --- | --- | --- | --- |
| Laydown 0 | yes | 0.484050 | 0.999975 | 0.042788 | 24.36 s |
| Laydown 1 | yes | 0.999975 | 1.000025 | 0.061561 | 23.47 s |

Values are un-clamped projected triangle-union coverage divided by the matching flat
reference. Slightly greater than one is retained. Dataset verification passed: two
valid attempts, zero invalid attempts, no missing/corrupt references or orphan files.
The dataset is approximately 20 MB with compressed privileged cloth traces.

Collector wall time was 50.05 s, approximately 144 actions/hour for this two-action
sample. This excludes process startup and original reset generation; it is not a
campaign-scale throughput estimate. Dataset provenance records clean executed code
`96fd70836a1cc02dddc707d4886665ccdafe1a20`, dependency-lock hash, full configuration,
package versions, GPU and driver. Subsequent changes added tests/documentation only.

The pair used by the successful first action was fixed before the separate matched
primitive comparison. Each primitive restored the exact same snapshot; the final
restore also matched. Both executions were valid:

| Primitive | Initial coverage | Settled final coverage | Gain |
| --- | --- | --- | --- |
| Controlled laydown | 0.484050 | 0.999975 | +0.515925 |
| Retained-grasp fling candidate | 0.484050 | 0.596950 | +0.112900 |

This is one favorable pair on one fixture, not a statistical primitive qualification.
The dynamic candidate did not outperform the baseline and is not ready for broad
fling training collection on this evidence.

## Verification and fixes grounded in actual failures

- **183 CPU tests passed.** Ruff lint/format and whitespace checks passed.
- **Two integrated RTX 5080 tests passed** in 52.39 s: exact nonflat observation and
  same-step restore; scene/attachment/release/reset lifecycle.
- GPU attachment endpoint error was approximately 2.95e-8 m; release preserved
  measured velocity. Control commands and contact-physics impulses were audited separately.
- Exact-mesh, two-sided depth rendering matched physics rays to a maximum sampled
  error of approximately 6.3 micrometres on the integrated folded fixture.
- A fresh built wheel verified the actual dataset and observation chain in a new
  environment containing neither Genesis nor Torch. Offline learning/data imports
  do not initialize simulation.
- Fold seed 11 and single-point drop seed 11 passed current runtime settling checks.
  No diverse-bunch reset distribution or held-out cloth generalization is qualified.

The implementation exposed and corrected stale first-frame/restore rendering,
visual-mesh smoothing and one-sided depth, foreground selection on the table,
sharp-crease reset impulses, graph-distance over-tautening, missing table friction,
nanometre placement roundoff, swallowed timeouts and fabricated terminal observations.
Failed debug runs remain local; invalid outcomes are recorded and are not valid rewards.

Numerical comparison kept material, trajectory, grasp and damping unchanged. Full-lift
peak strains at 16/32/64 stretch sweeps were 0.0963/0.0666/0.0320. Four/eight sweeps
crossed the 0.10 guard early. Raising sweep count alone did not make resets settle;
continuous crease geometry and explicit contact friction were separately necessary.

## Next gate

Before a broad training campaign, qualify more favorable-grasp fixtures, random-drop
seeds and bunch families; compare resolution/timestep sensitivity and the contact
approximation; improve the dynamic candidate if flings remain the intended primitive.
Then freeze reset-family splits and collect a bounded pilot dataset. Learned RGB
grasp selection, held-out policy improvement, cross-cloth evaluation, HF releases and
physical robot adaptation remain later milestones. See `milestone.md` for that boundary.
