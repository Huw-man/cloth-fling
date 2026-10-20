# Two-action image-grasp integration pilot

This experiment connects the existing transformed-RGB uniform feasible selector,
native assisted attachment and user-frozen post-grasp laydown. It uses exactly one
archived reference fixture and at most two actions with RNG seed 100. There is one
restore at episode start and no resets, replacement grasps or rerolls thereafter.
It tests execution continuity, not unfolding effectiveness or learned behavior.

From the project root, using the locked root environment with simulation/video
extras and the exclusive RTX 5080 lease:

```sh
UV_CACHE_DIR=/tmp/cloth-fling-uv-cache .venv/bin/python -m cloth_fling.reference_pilot
```

The current v4 default output is `runs/image-grasp-laydown-pilot-centered-drag-v4-20261019/`.
All previous outputs remain terminal and untouched. The runner claims
this directory before opening a scene and refuses any existing output, including
an interrupted, failed or successful run. No resume or automatic execution retry
exists. `--output` and `--fixture` select artifact locations; the input fixture must
match the exact archived metadata and native state hashes. These location options
do not authorize another experiment or extend the two-action budget.

The one-hour ceiling includes scene startup, collection, verification and replay.
Startup and replay each have a 180-second deadline; actions retain the frozen
35-simulated/180-wall-second bounds. Rendering consumes saved geometry without
physics execution and uses wide overhead/oblique cameras. No viewer is opened.

Each selection comes from the current observation. The runner resolves its two
pixels read-only using the native attachment resolver, then sorts only execution
endpoints by world Y. Original pixels, map encoding, resolved IDs and execution
permutation are retained. This ordering does not qualify arbitrary orientations.
After one attach, the frozen controller adopts the same two pins. Its physics,
trajectory, source hashes and observational placement policy remain unchanged.

Dataset intent and a separate immutable action claim precede acquisition. Native
states are saved before acquisition, after acquisition and through the existing
controller's phase/failure/termination capture. Independent cleanup certificates
are written before outcome/report serialization, including acquisition failures.
Invalid or uncertain actions end the episode. Ordinary placement-quality and
settling shortfalls remain observations; numerical/domain/deformation, ownership,
deadline, release and artifact regressions stop execution.

The report separates integration success, signed coverage changes and material
strain. Success requires two completed valid actions, verified native cleanup,
exact first-after/second-before state and observation continuity, dataset integrity
and complete replay generation. A single completed action is incomplete evidence.
There is no minimum coverage gain. The known friction timestep discrepancy and
unqualified material strain remain limitations; assisted pins do not establish
physical pinch grasping.

The dataset, action claims, source fixture, restoration certificate, cleanup and
report remain under the output directory. The replay is:

`runs/image-grasp-laydown-pilot-centered-drag-v4-20261019/replay/combined.mp4`

Fold generators, broader reset distributions, training, publication, cloud and
hardware work are deferred. All historical campaigns remain terminal.

## V4 measured centering correction

The v3 development episode completed the longer drags but failed to center the
folded cloth: final footprint centers were (.644077, −.028072) and
(.696095, −.104403) m. The actual drags increased instantaneous projected coverage
by .037200 and .053433, while final coverage was .419850 and .423200. Integration
passed; placement did not meet the centering goal. Preserve its 105.808-second
run in `runs/image-grasp-laydown-pilot-drag-v3-20261019/`.

One correction estimates trailing extent from measured airborne vertical drop,
using factor .43 calibrated on those two development outcomes. It also shifts
the diagonal smoothly in Y to align the observed hanging footprint. Record this
privileged execution geometry and actual coefficients, with no leakage into the
RGB selector. During the ensuing full 12 cm drag, Y and Z remain fixed.

The default `--budget-predecessor` points to the terminal v3 report. Subtract its
charge from the same one-hour ceiling; uncertain or exhausted predecessors cannot
open a scene. The additional check remains at most two evolving actions with
the same fixture/RNG and no resets/rerolls. This single correction is development
evidence, not general or held-out placement qualification.

## Historical V3 longer drag and nominal placement

The current user-requested revision uses a continuous 12 cm / two-second drag,
one-fifth of the cloth length. The diagonal ends at X=.73 m before dragging to
X=.85 m, allowing .30 m of trailing cloth to place its nominal center at X=.55 m.
Y and height stay fixed during the drag. This replaces the adaptive tug's early
slack/body-slip stops, while preserving all physical execution safeguards.
Scene, physics, fixture, RNG 100, two-action/no-reset limits and one-hour ceiling
are unchanged. No historical run is resumed or extended.

The report measures coverage before/after the drag separately from full-action
coverage change. It also records the final cloth footprint center, tabletop
center (.55, 0) m and the X/Y centering error. Integration requires the full 12 cm
target displacement and bounded continuous commands; placement quality and strain
remain separate measurements.

## Historical corrected v2 repeat result

The single authorized repeat completed on the RTX 5080 in **97.480 wall seconds**,
including saved-state replay generation, on integrated commit `03787f0`. Both
actions completed with exact state/observation continuity, native ownership and
release/cleanup verification. This time the recorded tug entry/exit commands also
passed continuity and kinematic checks. Both tugs moved forward 2 mm while keeping
each pin's Y and height fixed. The largest lateral target error was 6.94e−18 m;
the largest sampled tug speed/acceleration were .018256 m/s and .504 m/s².

The exact original fixture and RNG 100 were reused. The first selected pair and
pixels matched the original run exactly. The second pair was selected from the
corrected first action's resulting image, without any reset or substitution.

| Measurement | Action 0 | Action 1 |
| --- | --- | --- |
| Selected native IDs in execution order | 1568, 1914 | 132, 1204 |
| Incoming simulation time, s | 0 | 20.04 |
| Action simulation duration, s | 20.04 | 32.34 |
| Coverage before → after | .993333 → .449317 | .449317 → .625150 |
| Signed coverage change | −.544017 | +.175833 |
| Whole-cloth airborne hold, s | 3 | 3 |
| Minimum airborne clearance, m | .147586 | .117014 |
| Maximum pin tracking error, m | 4.10e−8 | 4.15e−8 |
| Final whole-cloth RMS speed, m/s | .002803 | .002562 |
| Settled within the fixed allowance | Yes | Yes |
| Final tabletop containment | 100% | 100% |
| Peak local material edge ratio | 1.564957 | 1.661032 |

The discontinuity is corrected; random grasp quality remains unqualified. The
second action improved coverage, but the episode still ended below its nearly
flat initial coverage. Material deformation remains substantial and unqualified.
The scene, contact parameters, diagonal geometry and timing limits were unchanged.

All 514 CPU tests passed (eight GPU tests deselected); 33 focused tests passed on
integrated main. The new saved-command audit rejects the original pilot's two
jumps. Independent native-state checks found finite states, consistent ownership,
zero release-velocity error, no positive friction work and no friction on pins.
All three replay files fully decoded: 527 frames, 10 fps, 52.7 seconds. Sampled
frames were inspected without a viewer pane. The GPU lease was released and no
compute jobs remained. This repeat is terminal; no rerolls or further episodes ran.

The report, native log, verification, independent cleanup certificates and dataset
are in `runs/image-grasp-laydown-pilot-tugfix-v2-20261019/`. CPU logs and the separate
assessment rejecting the untouched original run are preserved under
`runs/tug-frame-v2-verification/`.

## Historical v1 result and motion-continuity correction

The following original integration verdict checked state chaining and cleanup but
missed a tug-coordinate defect. Each tug added the world midpoint Y twice,
commanding lateral jumps of 47.24 mm and 106.44 mm. The corrected saved-command
assessment rejects both. Preserve the original run, including its report; its
claim of motion success is superseded. The v2 repeat fixes only this coordinate
bug and adds a motion-continuity requirement, with the same fixture, RNG and
two-action/no-reset limits. The second selected pair can differ because it is
selected from the corrected first action's actual result.

The single authorized episode completed on the RTX 5080 in **80.859 wall seconds**,
including replay generation, on integrated commit `152333f`. Integration passed:
both actions completed with exact observation/state continuity, no intervening
reset, intact native ownership, zero release-velocity error and certified cleanup.
The archived fixture hash matched exactly. The output directory is terminal; no
additional episode, retry or extension is authorized by this result.

| Measurement | Action 0 | Action 1 |
| --- | --- | --- |
| Selected native IDs in execution order | 1568, 1914 | 418, 2308 |
| Incoming simulation time, s | 0 | 19.94 |
| Action simulation duration, s | 19.94 | 21.90 |
| Coverage before → after | .993333 → .522433 | .522433 → .293817 |
| Signed coverage change | −.470900 | −.228617 |
| Whole-cloth airborne hold, s | 3 | 3 |
| Minimum airborne clearance, m | .150844 | .163654 |
| Maximum pin tracking error, m | 4.10e−8 | 4.11e−8 |
| Final whole-cloth RMS speed, m/s | .002720 | .148362 |
| Settled within the fixed allowance | Yes | No |
| Final tabletop containment | 99.57% | 100% |
| Peak local material edge ratio | 2.018889 | 2.184940 |

Both actions reduced coverage and recorded placement-quality violations. Action 1
also recorded incomplete settling. They completed under the authorized
demonstration policy; these outcomes do not qualify useful grasp selection or
unfolding. Material deformation remains substantial and unqualified. Contact work
was dissipative, with no positive friction work or friction applied to pins.
Contact totals in action evidence are cumulative across the episode; the separate
native artifact audit records their per-action increments.

Verification: 506 CPU tests passed (eight GPU tests deselected); 45 focused tests
passed on integrated main. The dataset verifier and independent saved-state
ownership/finite-state audit passed. All three replay files fully decoded: 421
frames at 10 fps, 42.1 seconds. Saved frames were inspected without opening a
viewer pane. The GPU lease was released and no compute jobs remained.

Evidence includes `pilot-report.json`, `verification.json`,
`native-artifact-audit.json`, `native.log`, independent cleanup certificates and
the dataset under `runs/image-grasp-laydown-pilot-20261019/`.
