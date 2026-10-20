# Reference towel environment

This independent Genesis 1.4.4 profile uses the supplied towel physics and controlled
placement, two ideal **single-vertex hard pins**, and a learning-ready FlingBot-style
grasp representation. It does not implement FlingBot's dynamic fling, train a policy,
or claim the paper's performance. Milestone 3 remains terminal and unchanged.

## Run and inspect

Use the existing root environment with simulation extras and the RTX 5080 exclusive
lease. The coordinator's bounded command acquires `/tmp/cloth-fling-rtx5080.lock`:

```bash
UV_CACHE_DIR=/tmp/cloth-fling-uv-cache uv run --locked --extra sim --extra video \
  cloth-fling reference qualify --output runs/reference-laydown-v1
```

Choose a fresh output directory. Existing runs cannot resume, acquire more seeds,
retry uncertain actions or extend budgets. The command records failure evidence and
returns nonzero when qualification is incomplete. It allows at most six diagnostic
actions and two episodes of at most three actions, 35 simulated/180 wall seconds per
action, and one hour total wall time. Scene preparation and contact work are charged
separately within that total. A failed physical prerequisite stops dependent work.
The current frozen runner performs one favorable-grasp diagnostic; the six-action
ceiling does not authorize automatic tuning or retries.

For an explicit static/reset inspection (not motion qualification):

```bash
cloth-fling doctor --profile reference --probe
cloth-fling resets --profile reference --kind authored --output runs/reference-reset
```

`configs/reference-laydown-v1.json` is the complete reference configuration. Passing
it through `--config` selects the reference backend. Motion/episode campaigns use
`reference qualify`; the legacy `diagnose` and `collect` commands intentionally reject
this profile rather than using square-cloth trajectories accidentally.

## Physics and reference differences

The authored towel is 0.60 × 0.40 m before XY contraction, surface area 0.24 m²,
surface density 0.5 kg/m² and nominal mass 0.12 kg. Visual/authored thickness is
2 mm; the 10 mm PBD particle collision diameter is separate. Native remeshing may
change surface area and hence discretized mass; archive native area, particles and
triangles instead of assuming the original grid or reference particle count survives.
The table is 0.80 × 1.00 m, centered at (0.55, 0), with surface z=0.30 m.

The frozen wrinkle adaptation is the sum of `sin(2π·3·(u+v)+phase0)` and
`sin(2π·3·(u−v)+phase1)` for normalized centered material coordinates. Two phases
come from NumPy's default RNG with seed 7. Subtract the sampled mean, normalize the
maximum absolute relief to 5 mm, and solve a uniform XY contraction for triangle
area 0.24 m². The original reference omitted exact mode weights/phases, so this is
an explicit adaptation. The authored/rest/native geometries have distinct identities.

PBD runs at 1 kHz, one substep, 128 stretch iterations, compliance 1e-7 and relaxation
0.3; one bending iteration, compliance 1e-5 and relaxation zero; no air drag. These
are engineering settings, not a conversion from IPC elasticity. The reference split
contact operator reads native velocities before/after each step, derives normal
velocity correction including gravity, and reduces horizontal speed by at most
0.45 times that correction. Only free particles within 1 µm of z=0.305 and inside
the table footprint qualify. This profile never installs native-v2 tangential contact.

Only selected fixed particles receive TCP commands. Each 50 Hz minimum-jerk waypoint
is linearly interpolated across twenty native steps. Native ownership, position
tracking (1 µm), and velocity-preserving release (1e-4 m/s) are checked. No robot,
spring, mesh teleportation during actions, or orientation-controlled patch is used.

## Motion and qualification

The policy chooses grasps only. The fixed sequence is attach .2 s; lift 2 s to
midpoint x=.10,z=.95; bounded adaptive tauten ≤20 s; airborne hold 3 s; laydown 2 s
to x=.70,z=.31; drag .8 s to x=.80; release .2 s; free settle ≤5 s. Initial midpoint
Y is retained, and lift aligns the horizontal grasp span along Y.

Tautening uses 65 barycentrically mapped material samples between the selected native
face centers. Increments add at most 2 cm total span, with ≤.02 m/s peak outward
speed per TCP, minimum .4 s duration, and endpoint holds ≤1 s. The reference tautness
and settling conditions use five consecutive 50 Hz samples. Span/duration bounds
proceed to placement; a tautness proxy is not a material-fidelity certificate.

Small strain is recorded without the old 10% brake. Edge ratio ≥3, nonfinite state,
solver-domain escape, solver failure or ownership loss stops motion. This gross
edge-ratio limit is an engineering default. Keep measured stretch in every report.

Qualification requires native restoration/rendering identity, a resting-slide contact
coupon, then complete controlled placement: three seconds of positive whole-cloth
clearance, retained pins through drag, release speed ≤.01 m/s, all final vertices on
the tabletop footprint, settled velocity, and certified release/cleanup. Only then
are two uniform-valid spatial-action episodes attempted. Each must complete two
valid consecutive actions; a feasible third may run. Evolved cloth is never reset
inside an episode. Both episodes descend from one authored fixture and therefore
provide continuity evidence, not diverse-reset generalization.

## Future learned grasps

`build_action_batch` returns 96 RGB tensors (12 rotations × 8 scales), each 64 × 64,
plus separate feasibility masks and inverse transforms. Angles are evenly spaced
from −90° to 90° and scales from 1 to 2.75. An RGB-derived square crop puts the
longest foreground extent at two-thirds of the unscaled image. Nearest-pixel
sampling and a fixed canonical 32-pixel grasp width are explicit adaptation choices.
Inverse transforms decode center/angle/scale into two original-image grasp pixels.
Depth is used only for execution feasibility, never placed in predictor RGB.
Final visible-vertex resolution remains authoritative and may reject a candidate.

`ValuePredictor` accepts only uint8 RGB `[96,64,64,3]`, returning finite values
`[96,64,64]`. `choose_action` masks infeasible candidates and takes the maximum;
without a predictor it samples uniformly with an explicit RNG. No neural network,
optimizer, checkpoint or training loop is included.

`learning_samples(dataset_dir)` verifies the dataset and yields only RGB, feasible
mask, encoded executed action, signed immediate reward and ancestry for valid settled
transitions. Invalid/unsettled attempts remain archived but are excluded from this
training view. The learning target is immediate normalized delta coverage, with no
bootstrapping. Coverage is projected native-triangle union on a 2 mm grid over
x=.15..95,y=−.40..40, normalized by the flat-reference union. Negative rewards and
coverage above one are retained. Geometry-based rewards differ from the paper's
visual-mask labels.

Parquet schema 1 remains unchanged; new action metadata, masks, commands, native
velocities, phase traces, topology and release checks live in hashed support bundles.
Schema-2 physical snapshots bind profile/configuration/runtime/topology identities.
Per-action cleanup certificates are written before report generation. All descendants
of the authored fixture retain the `development` split and `reference-authored-v1`
reset group. No training or evaluation split is fabricated from repeated resets.

See [FlingBot §§3.2–3.4](https://proceedings.mlr.press/v164/ha22a/ha22a.pdf) for the
prescribed-motion / learned-grasp distinction and spatial value-map supervision.

## Current result

The first motion qualification stopped during drag after successful whole-cloth lift
and airborne hold. Placement and continuity remain unqualified. See the
[measured results and failure evidence](reference-results.md). `--previous` supports
only terminal setup corrections and carries every prior budget charge; a predecessor
with motion attempts is rejected. Do not use new directories to extend this run.
