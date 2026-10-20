# Cloth-fling first milestone

## Genesis 1.4.4 migration completed

The user requested the latest stable simulator before further simulation work.
Genesis 1.4.4 (2026-10-11, upstream a649105) is integrated on local main;
the root `.venv` is upgraded. Separate Herdr worktrees delivered runtime/contact,
rendering/replay and release integration. 323 CPU tests, eight GPU tests and all
three bounded contact coupon profiles passed. See `docs/genesis144-migration.md`.
GPU leases are free, with no jobs remaining.

Preserve milestone 2 results as historical Genesis 0.4.3 evidence; new work must
not inherit their pass status. Use fresh schema-2 states and the native-v2 profile.
Before resuming episode collection, re-establish favorable-grasp primitive behavior
on fresh 1.4.4 fixtures. Migration checks do not qualify old motion primitives,
diverse-bunch reset distributions or learned policy behavior. No training,
publication, cloud or hardware work was part of this migration.

## Active milestone 3

The user authorized fresh Genesis 1.4.4 laydown qualification and a bounded paired
pilot on one synthetic 0.50 m square cloth. Milestone 2 results remain historical.
The complete stage/gate contract is in `docs/milestone3.md`.

The first bounded run stopped at reset preparation on 2026-10-19: development
6/6, held-out 5/6 after all twelve held-out candidates. The frozen manifest in
`runs/milestone3/` is terminal (`pool_limitation`). Qualification, continuity,
fling diagnostics and pilot behavior are not evaluated. Existing benchmark
prepare/qualify/pilot commands must return this stop offline, without opening a
scene or consuming additional seeds/actions. See `docs/milestone3-results.md`.

Three new Herdr worktrees own fixtures/manifest/reporting, primitive diagnostics,
and collection metadata/verification/recordings. The coordinator owns interfaces,
CLI, integration, exclusive RTX 5080 lease, and execution. Integrate reviewed
commits into local main and remove merged worktrees after preserving evidence.

Freeze fresh schema-2 folds 11/17/23 (development diagnostics only) and the first
six qualifying drops from each seed pool 1000–1011 and 2000–2011 before tuning.
Use only pre-action reset acceptance; retain every failure. Stop if either pool
cannot supply six. No physics/material/numerical/appearance/mesh/workspace changes.

Limits: 24 drop preparations, 26 laydown diagnostics (including at most one measured
trajectory revision), six frozen fling diagnostics, four two-action continuity
episodes, 120 pilot attempts, six hours total simulator-run wall. Each action
retains 30 simulated/120 wall seconds and existing settling/strain/clearance bounds.
Two valid consecutive-action episodes and laydown qualification precede pilot.
Pilot is 12 frozen drops × two selectors × at most five actions, development first,
held-out once, exact paired state, no resets within episodes. Preserve negative
gains and failures. Stop dependent work for restoration/chain/artifact/contact or
rendering regressions. Budget exhaustion is incomplete, never an extension.

No training, publication, cloud, hardware or broader collection is authorized.

## Historical milestone 2 scope

The user approved implementation of visible debug grasps, independently configured
matte desk appearance, native table-contact friction qualification, and whole-cloth
clearance before placement. The approved scope is recorded in `docs/milestone2.md`.
This ownership supersedes the historical stage ownership below:

- Contact agent: backend/contact/native-contact, contact diagnostics, SceneConfig
  contact fields, contact profiles/tests and contact documentation.
- Visual agent: appearance/debug rendering/replay modules, their tests and viewer docs.
- Clearance agent: primitive execution, clearance/callback tests and primitive docs.
- Evidence agent: CLI, collection/diagnostic recording integration, provenance,
  packaging/video dependencies, integration tests and user documentation.
- Coordinator: integration review, this file, milestone reports, bounded final host checks.

Use separate Herdr worktrees and commits. GPU access is a coordinator-assigned lease:
one scene at a time. Native contact qualification precedes primitive qualification;
stop dependent campaigns when its physical acceptance gate fails. Preserve failed
evidence. Do not silently change damping, thresholds or material values to pass.
No training, broad collection, publication, paid compute or hardware execution.

Build a standalone Python package, robotless Genesis scene, deterministic assisted
attachment, retained-grasp laydown/fling primitives, bounded repeated-action collection,
and Parquet/image/state dataset tools. Training/model releases and publication are later.

## Coordination

- Work in your assigned Herdr worktree and branch. Commit your owned paths only.
- The coordinator integrates to main. Do not merge, push, publish, or start broad campaigns.
- Shared interfaces are in `cloth_fling.contracts`; request contract changes from coordinator.
- Keep Genesis imports lazy. Offline dataset tools must work without Genesis or torch.
- Use world metres with z up, image (u,v) top-left, u right and v down.
- Never reset between valid actions. Record invalid attempts and terminal next observations.
- Keep negative coverage gains. Do not treat hand paths or regression loss as behavior proof.
- Attached particles only may receive motion commands. Release must retain velocity.
  A separate, explicit table-contact physics adapter may apply dissipative tangential
  Coulomb impulses to supported free particles; this must never be hidden as control.
- Privileged geometry/depth/state are execution/evaluation artifacts, never learned RGB input.
- No arms, IK, pinch grasping, AWS, neighboring-checkout imports, or machine-specific paths.

## Runtime

The user explicitly authorizes the host NVIDIA RTX 5080 (16 GB). Sandbox device masking
does not establish a host failure. Use approved host execution for bounded GPU qualification
and short smoke tests. Coordinate GPU use with the coordinator; avoid concurrent scenes.
Do not disable agent sandbox/approval protections. Use `UV_CACHE_DIR=/tmp/cloth-fling-uv-cache`.

## Ownership

- Stage 0 agent: `src/cloth_fling/sim/backend.py`, `sim/mesh.py`, `sim/coverage.py`,
  `tests/test_sim*`, `tests/test_coverage*`, `docs/runtime.md`.
- Stage 1 agent: `src/cloth_fling/sim/primitives.py`, `diagnostics.py`, `selection.py`,
  `tests/test_primitives*`, `tests/test_selection*`, `docs/primitives.md`.
- Stage 2 agent: `src/cloth_fling/data/`, `collection.py`, `tests/test_data*`,
  `tests/test_collection*`, `docs/dataset.md`.
- Coordinator: package configuration, contracts, CLI, integration, lock, README.


## Reference towel profile (new authorized work)

The user approved a separate reference setup on Genesis 1.4.4: 0.60 × 0.40 m
wrinkled towel, two single-vertex hard pins, reference split tabletop friction,
and prescribed lift/tauten/hold/laydown/drag/release. The grasp action interface is
FlingBot-style transformed RGB value maps; network training and dynamic-fling
reproduction remain deferred. See `docs/reference-environment.md`.

This profile does not change or resume milestone 3. Its local qualification is
bounded to six diagnostics and two episodes of at most three actions, 35 simulated/
180 wall seconds per action and one hour total simulator wall. No physics/motion
tuning after inspecting qualification outcomes. All failed evidence is preserved.
The first motion reached the solver boundary during drag after successful full-cloth
airborne hold; placement is unqualified and continuity was not evaluated. See
`docs/reference-results.md`. Do not infer diverse-reset or learned-policy success.

## Leading-edge development campaign (new authorization)

The user superseded the reference profile's no-motion-tuning restriction only for
`reference_leading_v2`: at most 20 deterministic grid attempts, followed by three
frozen confirmations, within two hours simulator wall. Restore the exact archived
reference fixture and favorable native vertices before each attempt. Preserve all
physics, appearance, mesh, grasp and table settings. Independent native cleanup and
durable intent records are mandatory; uncertain/failed attempts cannot be resumed.
See `docs/reference-leading.md`. Previous reference and milestone 3 campaigns remain
terminal. No learning, varied-grasp evaluation, cloud, hardware or publication.

The leading-edge campaign is now complete: candidate 7 passed development and all
three frozen confirmations after eight development attempts. The trajectory is in
`configs/reference-leading-v2-frozen.json`; results are in
`docs/reference-leading-results.md`. Its fixture claim is terminal. No additional
motion, changed grasps or new campaign budgets follow from that local qualification.

## Progressive diagonal and contact campaign (new authorization)

The user approved the plan in `docs/reference-diagonal.md`: explicit native
position-level Coulomb table physics, followed only after contact qualification by
at most 12 diagonal trials and three frozen checks. Contact qualification has at
most 16 .5-s/60-wall-s coupons; the entire campaign, including rendering, has two
hours simulator wall. Actions retain 35 simulated/180 wall seconds. Preserve material,
mesh, table, appearance, favorable pins and historical terminal campaigns. Native
friction may apply bounded dissipative tangential position corrections to supported
free vertices as explicit physics. Diagnostic force fields belong only to coupons.
No training, varied grasps, broad collection, cloud, hardware or publication.

The first diagonal-v3 campaign is terminal and incomplete: ten contact cases passed;
the .5-second airborne diagnostic landed at .404 s and failed its no-contact gate.
No motion trials or confirmations ran. The airborne case now ends at .3 s and has
CPU regression coverage, but has not been rerun on GPU. Do not resume or extend the
closed campaign or infer full contact/motion qualification. See
`docs/reference-diagonal-results.md` and the preserved original source/artifacts.

The user's subsequent request to perform a fling with the new friction/laydown
authorized a separate bounded demonstration. Its corrected contact gate passed
13/14 executed cases, then failed above-threshold breakaway at half the timestep
(0.183 mm travel versus 56.787 mm at the normal timestep). No motion ran and the
one-shot native ledger is closed. Preserve this failure separately; do not resume
either terminal run or infer contact qualification. See
`docs/reference-diagonal-demo.md`.

After the disclosed contact failure, the user explicitly renewed the request for
a table-start motion demonstration using the updated friction. One separate
exploratory action ran at the normal timestep without claiming qualification or
reopening a campaign. It lifted, tautened, held airborne for 3 s and entered the
diagonal, then stopped at 10.18 s on `deposited_body_sliding`; cleanup passed.
The same physics, favorable pins, trajectory and execution checks were retained.
No tuning, retry or confirmation was performed. Preserve its terminal ledger and
failure under `runs/friction-motion-demo-20261019/`; see
`docs/friction-motion-demo.md`. Provide replay file paths without opening viewer
panes; the user handles remote terminal-browser viewing with its own `--ssh` flag.

The user subsequently removed placement-quality hard requirements for exploratory
simulation: continue while physical execution remains stable. Demonstration mode
now logs cloth drift/deposition/overhang/release-speed/settling violations rather
than aborting for them. Numerical, domain, gross-deformation, ownership, deadline,
release-velocity and cleanup stops remain. Qualification-mode defaults and all old
terminal evidence remain unchanged. One newly authorized bounded demonstration
may execute with this mode, using the same physics, fixture, pins and trajectory;
no tuning, retries or qualification claims. File handoffs must use project-relative
paths and must not open viewer panes.

That demonstration completed all phases through release and five seconds of
settling in 20.08 simulated seconds: .979767 coverage, full tabletop containment,
RMS speed .00005389 m/s, verified pins/release/cleanup. The tug controller found
the cloth already geometrically taut and added no pull. Drift reached 39.77 mm
and was logged without stopping. Material peak edge ratio was 1.736297; the known
timestep discrepancy and unqualified material fidelity remain. Its ledger is
terminal; see `docs/friction-motion-continued.md` and
`runs/friction-motion-demo-continued-20261019/`. Historical evidence stays intact.

## User-frozen post-grasp laydown

The user selected the completed continuation as the frozen laydown to hand off to
after grasp. `reference_post_grasp_laydown_v1` preserves the reviewed physics,
trajectory and observational placement policy. The API in
`cloth_fling.frozen_laydown` adopts the two existing native pins without attach,
reset, restore or pixel reselection, includes lift/tauten/hold, and owns stepping
through release/settling and independently certified cleanup. Native pair order is
negative-to-positive world Y, as in the reviewed reference grasp. See
`configs/reference-post-grasp-laydown-v1-frozen.json` and `docs/frozen-laydown.md`.
Configuration and controller hashes are frozen; changes require a new version.
This is user selection of development evidence, not new contact, varied-grasp or
learner qualification. Post-grasp lifecycle/target parity is checked offline; no
new native campaign is implied. Preserve every historical terminal run unchanged.

## Two-action image-grasp integration pilot (new authorization)

The user authorized the refined plan in `docs/image-grasp-pilot.md`: one exact
archived reference fixture, RNG 100 uniform feasible image grasps, at most two
actions without an intervening reset, and the unchanged frozen post-grasp laydown.
The total wall ceiling is one hour, with existing 35 simulated/180 wall-second
action bounds. Preserve failed/uncertain actions without retry or reroll. Placement
quality is observational; physical execution, ownership, cleanup and artifact
regressions stop. No new fold recipes, broad collection, training, cloud, hardware
or publication. Deliver project-relative replay paths without opening a viewer.

The authorized episode is now terminal and complete for integration only. Both
image-selected grasps entered the unchanged frozen motion with exact continuity
and verified native ownership/release/cleanup, within 80.859 wall seconds including
replay. Coverage fell .993333 → .522433 → .293817; the second action did not settle
within five seconds. Peak edge ratios were 2.018889 and 2.184940. These remain
negative placement/material evidence, not unfolding or policy qualification.
Preserve `runs/image-grasp-laydown-pilot-20261019/`; do not rerun or extend it.

## Tug-frame correction and bounded repeat (new authorization)

The user authorized correcting the final tug's duplicated midpoint-Y offset and
repeating the same image-grasp experiment once. The original pilot's integration
verdict omitted motion continuity: its tugs commanded unintended lateral jumps
of 47.24 mm and 106.44 mm. Preserve that archive and its original reports; the
corrected saved-command assessment rejects it. Do not treat it as motion success.

`reference_post_grasp_laydown_v2` freezes the coordinate correction, with identical
scene/physics, diagonal geometry, schedule and tug limits. V1 configuration bytes
remain historical; reproduce v1 code at commit `152333f`. The current API defaults
to v2 and verifies its new implementation hashes. See `docs/frozen-laydown.md`.
One new episode may use the same exact fixture, RNG 100 and at most two evolving
image-selected grasps, with no reset between actions, no reroll or uncertain retry.
Keep the one-hour total and 35 simulated/180 wall-second action bounds. Record
placement/settling failures and material strain separately, and require native
cleanup plus saved-command tug continuity for integration success. No tuning,
training, broad collection, cloud, hardware or viewer panes. Preserve the separate
repeat in `runs/image-grasp-laydown-pilot-tugfix-v2-20261019/`.

That repeat is now terminal and complete for this correction/integration slice:
both actions completed, both 2 mm tugs preserved Y (maximum error 6.94e−18 m),
both cloth states settled, and native ownership/release/cleanup and exact state
continuity passed. Coverage was .993333 → .449317 → .625150; peak local edge
ratios were 1.564957 and 1.661032. The 97.480 wall-second total includes replay.
All 514 CPU tests passed, and the saved-command audit rejects the original jumps.
No useful general grasp selection or material qualification follows; no further
episodes or retries are authorized by this completed run.

## Longer centered final drag (new authorization)

The user requested a final drag of about one-fifth/one-quarter of the 60 cm cloth
length, then specified that the final cloth should be centered on the workspace.
`reference_post_grasp_laydown_v3` uses a full continuous 12 cm / two-second drag:
move the diagonal endpoint to X=.73 m and finish the drag at X=.85 m, allowing
.30 m of trailing cloth to place its nominal center at X=.55 m. Target
cloth footprint center (.55, 0) m and measure its actual final X/Y offset.
Do not silently center the grasp point instead of the cloth. Preserve Y/Z during
the drag and the v2 coordinate correction. Sliding/tautness no longer truncate
the commanded drag; numerical/domain/gross-strain/ownership/deadline/release and
cleanup safeguards remain. All physics, contact settings and appearance stay fixed.

Use one fresh bounded two-action image-grasp demonstration with the same fixture
and RNG 100, within one hour total and 35 simulated/180 wall seconds per action.
No rerolls, uncertain retries, reset between actions, broader collection or
training. The full drag uses the existing two-second reservation; do not extend
the tautening allowance. Preserve every old configuration/run, and record the
new run separately in `runs/image-grasp-laydown-pilot-drag-v3-20261019/`. Provide
the replay path relative to project root without opening a viewer pane.

The v3 nominal full-length offset did not achieve the user's centering goal:
the two footprints ended 9.82 cm and 17.96 cm from the workspace center, despite
completed 12 cm drags. Preserve this separate development record; its integration
status is not placement success. One measured centering correction uses the
airborne hanging drop to estimate trailing extent (factor .43, calibrated on those
two cases), and the projected Y footprint center to align descent. This is an
explicit development geometry heuristic, not trained or held-out qualification.

`reference_post_grasp_laydown_v4` retains the full continuous 12 cm drag, fixed Y/Z
during drag, exact fixture/RNG and all physics. Only the diagonal endpoint is
computed from the measured hanging geometry. Perform at most one additional
two-action check, charging the v3 run's 105.808 wall seconds against the same
one-hour total via `--budget-predecessor`. This is not a resume or reroll of v3.
Preserve its configuration/run and use
`runs/image-grasp-laydown-pilot-centered-drag-v4-20261019/` for the correction.
No further centering revisions or episodes follow from this bounded check.

The v4 check is complete and terminal. Both full 12 cm drags passed continuity
and bounds, with settled final footprint centers 1.38 cm and 3.20 cm from the
workspace center. Drag-only coverage gains were +.110200 and +.043967, but the
full random-grasp episode still reduced coverage .993333 → .573267 → .418417.
Peak local edge ratios 1.684783 and 2.063633 remain material limitations. The v3
development plus v4 correction charged 192.471 wall seconds combined; no further
episodes or tuning follow from remaining budget. Native ownership, release,
cleanup, exact continuity, artifact/replay auditing and 527 CPU tests passed.

## User-approved baseline for the next training pilot

The user selected the centered v4 demonstration as the trajectory to save and
build upon for the end-to-end training pilot. The authoritative selection record
is `configs/end-to-end-pilot-baseline-v1.json`, referencing the unchanged frozen
`reference_post_grasp_laydown_v4` specification and exact reviewed native evidence
and replay hashes. See `docs/frozen-laydown.md`. Keep the 12 cm / two-second drag,
measured hanging-geometry centering rule, phase timing, physics and native handoff
fixed when building the next pilot. Save per-action measured coefficients rather
than replacing the controller with one demonstration's literal Cartesian path.

The intended next slice learns RGB-based grasp selection around this continuation,
with privileged geometry confined to execution/feasibility/evaluation and no reset
between valid actions. This baseline selection does not reopen completed runs or
start training/collection. Specify the small training-pilot protocol and bounded
budget before execution. User approval of this motion is separate from held-out,
learned-policy, material or physical pinch-grasp qualification; preserve the known
limitations and all historical failures.

## Bounded image-grasp learning pilot (new authorization)

The user authorized implementation and local execution of `docs/image-learning-pilot.md`:
21 finite-radius folded schema-2 fixtures, splits frozen by fold family, the unchanged
v4 continuation/physics, RGB-only immediate-value grasp learning, one random round,
one learner-state refresh and a three-action paired held-out comparison. The ceiling
is eight hours execution and 432 action attempts, including all failures; reserve two
hours from collection for fitting/evaluation/reporting. No fixture substitutions,
rerolls, uncertain retries, motion/physics tuning, cloud, hardware or publication.
Use a new `runs/image-learning-pilot-v1/` root. Historical campaigns remain terminal.
Source/fixture/dataset/checkpoint identities and stage/action intent are durable;
incomplete preparation or any ownership/restoration/contact/rendering/cleanup/artifact
regression stops dependent execution. Stable placement quality remains observational.
Training is authorized only within this fixed pilot. Do not extend the budget or
infer broad generalization, material fidelity or physical pinch-grasp qualification.
