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

The default output is `runs/image-grasp-laydown-pilot-20261019/`. The runner claims
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

`runs/image-grasp-laydown-pilot-20261019/replay/combined.mp4`

Fold generators, broader reset distributions, training, publication, cloud and
hardware work are deferred. All historical campaigns remain terminal.
