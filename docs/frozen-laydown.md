# Frozen laydown after grasp

`reference_post_grasp_laydown_v4` lengthens the final drag and centers the user-selected
continuation from `docs/friction-motion-continued.md`. The operator
configuration is `configs/reference-post-grasp-laydown-v4-frozen.json`; an identical
packaged copy lets installed code load it without a checkout or run directory.
The specification pins controller implementation hashes, the reference physical
configuration, phase schedule and nominal trajectory template. Measured geometry
determines each action's actual endpoints and coefficients, which are recorded
with the execution. Changes to the frozen controller require a new version.

## Approved training-pilot baseline

The user approved the centered v4 two-action demonstration as the motion baseline
for the next end-to-end training pilot. The selection record is
`configs/end-to-end-pilot-baseline-v1.json`. It pins the unchanged v4 configuration
and the exact reviewed report, verification, dataset manifest, fixture and replay
by SHA-256. This separate record supersedes earlier demonstration selections for
future pilot work without rewriting the configuration's historical provenance.

The reviewed replay is
`runs/image-grasp-laydown-pilot-centered-drag-v4-20261019/replay/combined.mp4`.
All paths in the selection record are relative to the project root. Existing run
directories remain terminal; selecting this baseline does not resume them.

The next slice builds the RGB-to-grasp-to-laydown loop around this fixed
continuation. Learning targets grasp selection; native hard-pin attachment,
measured centering and the full 12 cm drag remain execution components. Keep
privileged geometry out of learned RGB inputs, preserve evolving state between
valid actions, and retain signed coverage changes and failures. The training
protocol, reset/split choices and bounded execution budget were specified separately
in `docs/image-learning-pilot.md`. No training or new collection was performed by
the baseline freeze itself; the subsequent bounded learning experiment is recorded
in `docs/image-learning-pilot-results.md`.

This is a user-approved local motion, with the development evidence and material
limitations below. It does not confer held-out or learned-policy qualification.

## Handoff

The grasp stage hands over the live scene and its original `GraspAction` after
acquiring two native hard pins. `execute_frozen_laydown` adopts those pins without
calling `attach`, selecting new pixels, resetting, restoring or replacing the
cloth state. Simulation time continues from the incoming scene. The action's
pixels are retained solely as provenance; current native ownership defines the
held vertices.

Use the reference scene with `reference_position_coulomb_v1` at 1 ms. Exactly two
distinct native particles must already be fixed, with pin 0 on the negative-Y
side and pin 1 on the positive-Y side, matching the controller's fixed world-Y
spread convention. The caller must preserve that pair order when acquiring its
grasp. The handoff is intended after a table-start grasp, before lift; a moving
grasp transfer or arbitrary orientations have not been evaluated.

```python
from cloth_fling.data.storage import atomic_write, json_bytes
from cloth_fling.frozen_laydown import execute_frozen_laydown

# env already holds the two pins selected by grasp_action. run_dir is this action's
# artifact directory; the grasp controller stops stepping during the continuation.
def certify_cleanup(certificate):
    atomic_write(run_dir, "laydown-cleanup.json", json_bytes(certificate))

outcome = execute_frozen_laydown(
    env, grasp_action, cleanup_callback=certify_cleanup,
)
```

Precondition failures leave the existing grasp and scene untouched for the caller.
After a valid handoff, the laydown owns stepping, releases the pins at completion
or execution failure, and invokes the required durable cleanup callback before
returning an outcome. No physical command is automatically retried.

## Frozen motion

The continuation retains the same .2-second initial hold as the reviewed action,
now named `grasp_handoff`, followed by:

- Two-second lift to midpoint X=.10 m, Z=.95 m, preserving midpoint Y.
- Bounded tautening and three seconds of whole-cloth airborne hold.
- A six-second C2 diagonal toward an endpoint computed from the hanging cloth,
  ending at Z=.31 m, preserving the tautened span.
- One-second stabilization, then a continuous two-second C2 drag forward 12 cm
  at Z=.31 m. The drag is one-fifth of the 60 cm cloth length.
- One-second endpoint hold, velocity-preserving release, .2-second release hold
  and five seconds of unsupported settling.

The same 35 simulated / 180 wall-second action limits apply from the handoff.
Whole-cloth placement quality remains observational: drift, deposition, overhang,
release speed and settling are measured without aborting stable execution.
Numerical/domain/deformation/ownership/deadline and native release/cleanup stops
remain. This is the existing `demonstration` execution policy, now selected as the
frozen post-grasp continuation; historical qualification defaults are unchanged.

## V4 measured placement correction

The v3 nominal full-length placement completed its 12 cm drags but left the
folded cloth 9.82 cm and 17.96 cm off center. V4 uses the measured hanging cloth
instead of assuming a fully extended .60 m trailing body. After the airborne
hold, estimate the final grasp's X offset from the desired cloth center as .43
times the hanging vertical drop. Subtract .12 m for the diagonal endpoint. During
the diagonal, smoothly shift Y by the negative projected cloth bounding-box Y
center. The full two-second drag then holds that resulting pin Y/Z fixed.

The .43 factor is a single measured development correction from the two v3 cases,
not a general cloth model or learned selector. Record the measurement, estimated
offset and actual polynomial coefficients in each action. Check kinematic bounds
before executing the adapted segments; no free-particle relocation is permitted.
The six-second diagonal, two-second drag and complete phase reservation are
unchanged. V3 code/configuration remain reproducible at `a67963d`.

The bounded v4 check completed both 12 cm drags and settled with footprint centers
1.38 cm and 3.20 cm from (.55, 0) m. Drag-only coverage gains were +.110200 and
+.043967; full-action random grasps still reduced coverage overall. Native ownership,
release, cleanup and state continuity passed; peak local edge ratios were 1.684783
and 2.063633, so material fidelity remains unqualified. The v3 development case and
v4 correction used 192.471 wall seconds combined, within their shared hour. See
`docs/image-grasp-pilot.md`; neither run may be resumed or extended.

## Historical V3 final drag and placement target

The user requested a meaningful final table drag and a centered cloth afterward.
V3 replaces the millimetre adaptive tug with a full 12 cm drag over two seconds.
Body sliding and geometric tautness no longer shorten it. The numerical/domain,
gross-strain, ownership, deadline, release and cleanup safeguards remain active.
Placement-quality measurements remain observational; physics is unchanged.

The final leading edge is nominally at X=.85 m: .30 m beyond the tabletop center
for the .60 m cloth trailing behind it. Subtracting the .12 m drag places the
diagonal endpoint at X=.73 m. The placement target is the tabletop center (.55, 0) m; report
the actual final projected bounding-box center and its X/Y error. The target is
for the cloth footprint, not the grasp point, and is not a claim of exact centering
for arbitrary grasps or folded shapes.

The continuous quintic drag starts and ends at rest, preserves each pin's Y/Z,
and has analytic peaks .1125 m/s and .173205 m/s². It uses the same two seconds
already reserved for the old maximum tug, so the 14.6-second tautening allowance
and 35 simulated/180 wall-second action bounds remain unchanged. The saved-command
audit checks the full 12 cm was executed, as well as entry/exit continuity and
kinematics. The v1/v2 configuration bytes and previous run archives are unchanged;
reproduce v2 at commit `03787f0`.

## Historical V2 coordinate correction

The curve executor adds the held midpoint Y to its Y-relative segments. The old
tug supplied world Y instead, adding the offset twice at its first native substep.
V2 supplies zero relative Y for every tug increment. Its existing 1 mm forward
quintic therefore preserves each pin's Y and Z continuously, including off-center
grasps. Scene, friction, geometry, schedule, tug limits and placement policy are
unchanged. No motion tuning is included.

The v1 configuration and packaged copy retain their original bytes and hashes;
checkout commit `152333f` to reproduce the old controller. V2 froze the correction;
the current v3 API retains it with its updated controller hash. The original favorable demonstration executed no
tug, so it did not exercise the defect. The first image-grasp pilot did: its two
unintended Y jumps were 47.24 mm and 106.44 mm. Its recorded integration verdict
is superseded for motion continuity; its archive remains untouched.

Regression tests execute all 20 tug increments at the native 1 ms target cadence
for centered, positive-Y and both observed negative-Y grasp offsets. They check
the initial join, fixed per-pin Y/Z, total forward displacement and speed/
acceleration bounds. The pilot now independently assesses recorded tug targets,
including entry and exit, and rejects the original jumps. Its repeat is separate
evidence in `docs/image-grasp-pilot.md`.

The v2 repeat completed both image-selected actions with native cleanup and exact
state continuity. Both executed 2 mm tugs with at most 6.94e−18 m lateral target
error; both cloth states settled. Coverage was .993333 → .449317 → .625150, and
peak material edge ratios were 1.564957 and 1.661032. These results verify this
coordinate correction and local integration, not general grasp or material
qualification. The repeat remains terminal and separate from all v1 evidence.

## Historical v1 evidence and verification scope

The user approved one favorable-grasp native demonstration: .979767 final
coverage, full table containment, three seconds airborne, finite state, settled
cloth and verified cleanup. Its replay is:

`runs/friction-motion-demo-continued-20261019-review/combined.mp4`

The post-grasp adapter is verified offline for exact controller-target parity,
continued simulation time, native ownership adoption, no reattachment/reset,
invalid-handoff behavior and cleanup after an execution failure. The packaged
configuration is usable offline. No additional native run or confirmation is
claimed for the adapter. Varied-grasp and learned-policy evaluation remain
deferred, as do resolution of the known half-timestep friction discrepancy and
material-fidelity qualification (the demonstration's peak local edge ratio was
1.736297). Freezing this user-selected motion does not reopen any old campaign.

Verification passed 491 CPU tests (eight GPU tests deselected), source/artifact
hash checks, and an offline wheel build. The extracted wheel also passed the
post-grasp target-parity/lifecycle test independently of the source checkout.
