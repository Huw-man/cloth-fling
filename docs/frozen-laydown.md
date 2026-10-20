# Frozen laydown after grasp

`reference_post_grasp_laydown_v1` is the user-selected continuation from the
completed demonstration in `docs/friction-motion-continued.md`. The operator
configuration is `configs/reference-post-grasp-laydown-v1-frozen.json`; an identical
packaged copy lets installed code load it without a checkout or run directory.
The specification pins controller implementation hashes, the reference physical
configuration, exact trajectory coefficients/schedule, and the selected run,
outcome and replay hashes. Changes require a new version.

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
- The six-second C2 diagonal to X=.78 m, Z=.31 m, preserving the tautened span.
- One-second stabilization, then the existing adaptive tug capped at 2 cm. It may
  add no pull when the geometric slack target is already satisfied.
- One-second endpoint hold, velocity-preserving release, .2-second release hold
  and five seconds of unsupported settling.

The same 35 simulated / 180 wall-second action limits apply from the handoff.
Whole-cloth placement quality remains observational: drift, deposition, overhang,
release speed and settling are measured without aborting stable execution.
Numerical/domain/deformation/ownership/deadline and native release/cleanup stops
remain. This is the existing `demonstration` execution policy, now selected as the
frozen post-grasp continuation; historical qualification defaults are unchanged.

## Evidence and verification scope

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
