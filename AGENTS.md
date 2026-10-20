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
