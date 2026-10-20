# Progressive diagonal implementation and contact campaign — 2026-10-19

The position-level friction adapter and progressive diagonal/slack-taking-tug runner
are implemented. **The campaign is incomplete and terminal: ten contact checks
passed, then an invalid airborne-test setup stopped qualification. No diagonal
trials or confirmations ran.** The corrected airborne fixture is tested offline;
it has not been rerun on the GPU. No successful laydown or frozen trajectory is
claimed for this profile.

## Native sticking and sliding results

The new Genesis 1.4.4 scene-local contact law acts on tangential position and velocity
within the native normal-load Coulomb bound. It retains effective μs=μk=.45 and all
reference cloth/material/mesh/table/appearance/action-solver settings. Diagnostic
external loads are confined to coupons and normalized by native mass participation.

Measured coasting deceleration was **4.3285 and 4.3931 m/s²**, against expected
μg=4.4145 m/s². Below-threshold uniform loading produced **1.416 µm** maximum
95th-percentile drift. Below-threshold edge loading produced **50.66 and 38.51 µm**
unloaded-body drift. Above-threshold uniform loading produced acceleration
**2.2092 m/s²**, against expected 2.20725 m/s², and 44.61 mm displacement.
Above-threshold edge loads also broke away. This is evidence of finite sticking
and breakaway in the tested coupons, rather than permanent attachment.

Loads act from .1 to .3 s; drift is measured over that interval, excluding initial
flat-coupon relaxation. The signed displacement column is center-of-mass travel,
while the p95 drift column measures unloaded vertices for edge-load cases.

| Contact case | Result | Max unloaded p95 drift (mm) | Signed load-window COM travel (mm) |
| --- | --- | --- | --- |
| rest | passed | 0.000000 | 0.000 |
| coast_positive | passed | 0.000000 | 0.000 |
| coast_negative | passed | 0.000000 | -0.000 |
| uniform_sub | passed | 0.001416 | 0.000 |
| uniform_super | passed | 44.645391 | 44.607 |
| edge_sub_positive | passed | 0.050657 | 0.016 |
| edge_sub_negative | passed | 0.038514 | 0.012 |
| edge_super_positive | passed | 57.270840 | 56.769 |
| edge_super_negative | passed | 54.289207 | 53.771 |
| oblique_landing | passed | 4.289759 | 4.250 |
| airborne | failed setup assertion | 20.405471 | 20.356 |

All recorded cases respected the friction impulse bound, reported nonpositive
friction work within tolerance, retained finite native state and certified cleanup.
An independent saved-artifact audit checked native hashes and recomputed drift for
all eleven cases. These audit passes do not turn the final diagnostic into a pass.

## Why the campaign stopped

The executed airborne case started .8 m above support and ran for .5 seconds.
It legitimately reached the tabletop at **.404 seconds**. Its whole-run assertion
of zero table contacts therefore failed because the diagnostic no longer remained
airborne. This was an implementation mistake in the test fixture, not evidence that
friction acted on unsupported particles. The saved contact intervals show no table
contact through .4 seconds.

The repository now ends this case at **.3 seconds**, within the original .5-second
maximum. A CPU regression checks the semi-implicit ballistic margin at both native
timesteps; the existing native trace has .35794 m minimum clearance at .3 seconds.
The raw .5-second run, its failed assertion, original source and ledger are unchanged.
The correction is recorded separately in `airborne-fixture-correction.json`.
There was no retry, additional seed, changed coefficient, relaxed gate or extended
campaign. The exclusive claim stays terminal even with the corrected code.

The attached-particle exclusion, two half-timestep checks and two zero-friction
controls were not reached. The corrected airborne fixture also lacks a fresh GPU
qualification. Contact qualification consequently remains incomplete, and the
motion stage returns `contact_gate_stop` offline.

## Implemented motion, not executed evidence

The runner removes the high forward transfer. Its twelve frozen candidates use
endpoint X .78/.76/.74 m, each with durations 6/8/10/4 s, for one straight quintic
from x=.10,z=.95 to z=.31. Midpoint Y and grasp separation remain fixed.
The complete schedule reserves stabilization, a ≤2 cm adaptive forward tug,
endpoint hold, release and settling before assigning bounded tautening time.

Fixed material cohorts record their first settled landing locations. Each cohort
retains its own drift statistic, so later landings cannot dilute an earlier group's
sliding. The tug uses rest-to-rest 1 mm increments and stops on geometric slack
removal or incipient deposited-body slip. Coverage, progressive deposition, body
sticking, pin tracking, release and cleanup are all separate gates. See the
[implementation contract](reference-diagonal.md) for exact thresholds.

The new schema-2 fixture preserves the parent positions, velocities, free flags and
native mesh exactly, with a distinct contact/runtime identity and parent ancestry.
Old reference and milestone 3 qualification is never inherited. Earlier historical
campaigns and their terminal behavior are unchanged.

## Strain, budget and delivery

Peak local native edge-length ratio across the saved contact coupons was
**1.033199**. Material fidelity is not qualified; no trajectory strain result exists.

The native campaign used **11 of 16 contact reservations**, zero of twelve motion
trials and zero of three confirmations. Preparation/contact work was charged
**97.236 wall seconds**. Render-only replay is
accounted separately in the same `diagonal-budget.json`: 69.195 seconds,
for 166.431 seconds combined (under three minutes). The total remains within
the two-hour ceiling. No motion budget was consumed to work around the failed gate.

Ignored local artifacts remain under `runs/reference-diagonal-v3/`: original
configuration, source copies/hashes, ancestry, distinct native fixture, all eleven
coupon state/velocity/contact bundles, external-load definitions, cleanup
certificates, failure record, budget, execution log and offline audit. Wide overhead,
oblique and combined replays of the saved contact cases are under
`runs/reference-diagonal-v3-replay/`. They never step physics. There is no successful
trajectory replay to open.

Final CPU verification passed **470 tests**, with eight GPU tests deselected. The
actual host campaign supplies the native evidence above. Offline checks also cover
Coulomb sticking/breakaway, position/velocity consistency, impulse and energy bounds,
straight-path kinematics, reserved timing, fixed-cohort slip, tug termination,
explicit fixture ancestry, irreversible claims and terminal gate behavior.
