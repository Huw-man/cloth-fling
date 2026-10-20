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

The corrected v2 default output is `runs/image-grasp-laydown-pilot-tugfix-v2-20261019/`.
The original v1 output remains terminal and untouched. The runner claims
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

`runs/image-grasp-laydown-pilot-tugfix-v2-20261019/replay/combined.mp4`

Fold generators, broader reset distributions, training, publication, cloud and
hardware work are deferred. All historical campaigns remain terminal.

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
