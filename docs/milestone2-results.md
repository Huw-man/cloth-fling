# Historical milestone 2 evidence: Genesis 0.4.3

2026-10-19. The user requested migration to the latest stable Genesis before further
simulation work. These measurements are archived development evidence for **0.4.3
only**, not qualification of the upgraded runtime. No training, publication, paid
compute or hardware execution occurred.

Four Codex agents implemented contact, rendering, clearance and evidence stages in
separate Herdr worktrees. Their reviewed commits were merged into local main.

## Delivered and checked

- Native normal-impulse-bounded table friction, with separate table/self-contact
  coefficients and auditable cumulative impulse/work counters. Installed Genesis
  files were unchanged. Default .40/.30 and low/high sensitivity profiles passed
  bounded unforced coupon checks; these are engineering priors, not measured terry
  cloth on laminate coefficients. Actual desk calibration remains pending.
- Neutral gray matte table, independent appearance configuration, and red/blue
  10 mm debug grasp spheres. Native checks confirmed markers disappear on release,
  do not affect physics, and leave policy RGB/depth bitwise unchanged.
- Whole-cloth surface clearance of 20 mm held for 100 ms before placement, with
  bounded incremental lifting to 75 cm and unchanged strain/deadline guards.
  Protected fling swing/pullback stay level; descent begins during lowering.
- Optional native execution videos, reconstructed replay, contact CLI, provenance,
  and an explicitly experimental longer-laydown configuration.
- **284 CPU tests and three integrated RTX 5080 tests passed.** GPU tests took
  60.57 s. Ruff lint/format, lock and whitespace checks passed. A fresh wheel loaded
  historical/current datasets and replay trajectories without Genesis, Torch or
  video dependencies. GPU: RTX 5080, driver 595.91.07.

## Contact measurements

| Profile (static/kinetic) | Measured slide deceleration | Expected mu*g | Half-dt stopping-distance change |
| --- | --- | --- | --- |
| .40/.30 | 2.9454 m/s² | 2.9430 | 0.2405% |
| .25/.20 | 1.9637 m/s² | 1.9620 | 1.9343% |
| .60/.45 | 4.4106 m/s² | 4.4145 | 0.2523% |

Native microtests checked impulse bounds, friction energy loss, unchanged normal
response, zero attached/airborne friction, and logging isolation. Horizontal and
oblique flat-coupon drops showed no normal-energy increase. These initially
unstressed coupon tests do not establish energy behavior of arbitrarily folded cloth.

## Motion outcomes and stop conditions

The initial matched comparison used three frozen fold fixtures, seeds 11/17/23,
with the same reviewed pair within each fixture. Twelve single-action diagnostic
episodes produced seven valid and five invalid outcomes. They were collected as
diagnostic evidence, not a policy evaluation or a training release.

On seed 11, the old partial lift with the new contact model reached approximately
100% coverage. Fully lifting to 50 cm with the old 12 cm laydown + 6 cm drag reached
only 56.0%. This exposed inadequate placement travel rather than a reason to train
more or increase damping. A geometry-guided bounded check changed **only laydown
travel to 30 cm**, with drag still 6 cm:

| Fixture | Coverage before → after | Grasp height | Minimum held surface clearance | Peak edge-strain proxy | Action wall time |
| --- | --- | --- | --- | --- | --- |
| 11 | .484275 → .999950 | .500 m | .020107 m | .02889 | 37.28 s |
| 17, frozen validation | .483975 → .999900 | .500 m | .020021 m | .04169 | 36.07 s |

The 38 cm alternative was not run after 30 cm passed. The named experimental
profile preserves this setting without changing reference or legacy defaults.

The original fling's pullback deliberately descended 40 mm; this conflicted with
its protected-clearance phase. The v2 correction keeps that phase level. On fixture
11 the corrected motion maintained at least 48 mm clearance through swing/pullback,
then exceeded the unchanged 10% edge-strain ceiling during lowering. Fixture 17
also exceeded the ceiling during lowering. The earlier `premature_table_contact`
reason was a geometric guard at 19.6 mm clearance, **not proof of actual contact**.

Fixture 23's selected pair exceeded the strain ceiling during the initial lift
under both reference and clearance trajectories. This failure preceded the changed
placement phase; thresholds were not relaxed and the unchanged early phase was not
repeated in the followup.

A separate CPU visibility/material review identified one accessible material edge
with near-corner and inset pairs. Three additional bounded executions (one laydown,
two flings) also exceeded the strain guard. Thus grasp suitability was checked
separately, but full-fling behavior remains unqualified.

Across the 12-action comparison, four placement checks and three material-edge
checks, all **19 attempts** were retained: nine valid and ten invalid. All three
datasets verified. External videos/reports intentionally remain outside their
dataset manifests and are reported as external/orphan files by the verifier.
The 12-action mixed diagnostic took 357.6 s including fixture preparation, restores,
storage and selected native video capture; this is not campaign throughput.

The three-fixture primitive gate did not pass, so the planned new repeated-action
collection and broad collection were deferred. Two fixed-case successes are not
held-out policy improvement, general bunch unfolding or cross-cloth generalization.

## Local evidence and revisions

All run artifacts remain ignored under `runs/milestone2/`:

- `summary.json`: aggregate checks, outcomes and artifact hashes.
- `contact/`: verified native/low/high reports, full-state traces and native microtests.
- `visuals/`: native rendering isolation and bounded historical replay checks.
- `primitive-qualification/`: original comparison at clean revision
  `382d2a51db03978a75cab98fd8866ab38c634f81`.
- `placement-review/` and `edge-grasp-review/`: corrections at clean revision
  `dbaa6278527a1e8a56f1094ff59e3cf11df71740`.
- `review/index.html`: native videos, including the successful longer laydown.
- `final-gpu-tests.log`: integrated regressions on revision
  `02838ab7610eb9c0f1113c7f508cdb4e1653225c`.

Upgrade qualification must create new runtime identities and new evidence. Preserve
these historical artifacts and do not reuse their pass status for Genesis 1.4.4.
