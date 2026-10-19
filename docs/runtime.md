# Stage 0 Genesis runtime

Stage 0 owns the scene/mesh/coverage only. All attachments are explicit assistance,
one particle per grasp. Runtime function is not contact-grasp or primitive quality
evidence. No robot, training, cloud compute, or publication is involved.

## Integration interface

`GenesisBackend(config, backend="gpu", headless=True)` lazily imports Genesis.
Importing backend/mesh/coverage initializes neither Genesis nor torch. Explicit
CUDA selection rejects silent CPU fallback. The backend alone calls scene.step;
consumers call `set_attachment_targets` then `step`. Four solver substeps use the
velocity `(target-current)/sim_dt`; unattached particle state is never commanded
outside reset/restore. A target is for the next step; subsequent steps hold it.
Release flips the free flag without zeroing velocity. GJK rigid collision mode is
enabled because the PBD substep is 0.5 ms (outer dt 2 ms). `attached_ids` is an ordered
tuple. `close()` destroys this scene, not the shared process-global Genesis runtime.

Extra arrays: `rest_positions` (N,3), `faces` (F,3), `edges` (E,2), and
`rest_edge_lengths` (E,) use Genesis's runtime particle order. Snapshot contains
positions, velocities, free flags, topology, rest positions, scene/config hashes,
schema version, calibration JSON, simulated time, and an integrity hash. Values
are strings, scalars, or non-object arrays, compatible with NPZ allow_pickle=False.
Restore validates everything before writes, restores position/velocity/time,
rejects saved free flags containing any attached particles, clears ALL current
attachment flags and pending targets, and checks device readback. Static table
and material/rest constraints are defined by scene identity;
this is not an arbitrary Genesis checkpoint or a cross-version restart format.
`reset_metadata` is a JSON dictionary, also saved as `reset_metadata_json`, that
records preparation kind, seed, actual target/profile and acceptance/failure.

Genesis 0.4.3 unconditionally isotropically remeshes PBD cloth using
`PBDOptions.particle_size` as target edge length as well as collision diameter.
We set it to `config.mesh_spacing_m`. The original 51x51/5000-face square becomes
2654 runtime particles and 5106 faces in the tested environment. The backend
adopts that runtime ordering/topology and verifies bounds, planar rest shape,
and 0.25 m2 surface area. It does not assume OBJ vertex IDs survive loading.

## Calibration agreement for Stage 1

The actual rasterizer is **perspective**, overhead at world (0,0,1.8), looking
down -z, with image right +x and image down -y. Observation.calibration has:

- `model="perspective"`
- `intrinsics`: JSON 3x3 matrix K from Genesis camera
- `camera_to_world`: JSON 4x4 matrix using OpenCV camera axes right/down/forward
- `image_size=[width,height]`
- `pixel_center_offset=0.5`
- `depth_units="metres"`, `depth_convention="positive_camera_z"`, `world_up="+z"`

Integer array sample [v,u] backprojects as
`q = depth[v,u] * inv(K) @ [u+0.5, v+0.5, 1]`, then
`world = camera_to_world @ [q.x,q.y,q.z,1]`. To project a world point, invert
camera_to_world, apply K, divide by positive camera z, subtract 0.5.
There are no orthographic bounds. Stage 0 ray-tests the first cloth surface;
eligible picks are visible vertices of that triangle within 1.5 mesh spacings,
with Euclidean distance then particle ID tie-breaking. A nearer hidden layer
cannot win. Misses, coincident IDs, workspace and separation violations reject
before any attachment mutation.

The table is neutral dark gray (0.04,0.04,0.04); the cloth is saturated orange
(0.85,0.25,0.12). Use an RGB chroma test as well as brightness, as measured below.

Coverage is a boolean union of projected triangles on fixed 2.5 mm world cell
centres over the table, including overlap once and preserving concavities/holes.
Flat reference uses exactly the same mesh/grid; measurements outside the table
are clipped. Coverage can exceed 1 under stretch; negative action gains remain.

## Reset preparation and limits

All resets start from the declared rest mesh, zero velocities and no attachments.
`flat` keeps its centered orientation. `fold` uses a seeded planar rotation and a
reset-only half-fold with finite layer separation, then settles. `drop` samples
one seeded random material particle, raises it to `(0,0,h)` with h uniform in
[0.5,0.9] m over 2.5 s using quintic interpolation with zero endpoint velocity,
holds 0.5 s, and releases it. Only that attached particle receives commands during
the lift; full-state writes are restricted to initial reset/explicit restore.

Settling requires RMS **vector speed** below 0.01 m/s at every outer step for
0.5 simulated seconds, within a 10 s settling budget. Finite state, table XY
containment and nonnegative support-height tolerance are independently checked.
Failure raises `InvalidResetError`, records status/reason, preserves the failed
terminal state for observation, and clears all attachments. There is no flat
fallback, damping/friction adjustment, or relaxation of the acceptance gates.

**The random drop reset distribution is unqualified.** Tested final-preparation
seeds 0, 11 and 24 escaped the 1.2 m table during settling. Seed 24 was chosen by
an offline geometry check for a central held particle (rest XY approximately
[-0.01295,0.001715]); it also failed. This is evidence against assuming that
central acquisition alone fixes the drop reset. Default `kind="drop"` can raise;
the coordinator should explicitly choose a qualified flat/fold fixture for the
short integration smoke. Do not silently relabel a fold as a random drop.

Reset inputs are deterministic for a seed and mesh. GPU solver reductions can
vary at floating point precision; repeated reset observation IDs are not promised
bitwise identical. IDs hash actual state, time, RGB/depth and scene identity, are
stable for repeated observation without stepping, and reproduce exactly on the
tested free-state snapshot restore. An observation consumer gets copies.

## Tested runtime and commands

2026-10-19: own sim extra installed with
`UV_CACHE_DIR=/tmp/cloth-fling-uv-cache uv sync --extra sim --extra dev`.
Genesis 0.4.3 omits torch from wheel dependencies. Installed torch 2.11.0+cu128
using `uv pip install --python .venv/bin/python torch --index-url
https://download.pytorch.org/whl/cu128`. Coordinator is adding the locked CUDA
torch dependency/index to sim extra and owns uv.lock; Stage 0 does not modify
package configuration or commit the generated lock. No neighboring project
source or environment is imported.

Tested versions: Python 3.12, Genesis 0.4.3, torch 2.11.0+cu128, Quadrants 0.4.4,
NumPy 2.5.4, trimesh 4.12.2, pymeshlab 2025.7.post1, pytest 9.1.1.
Host nvidia-smi confirms RTX 5080 16303 MiB, driver 595.91.07, CUDA driver 13.2.
Sandbox nvidia-smi fails because devices are masked. Host qualification uses
approved elevated execution without changing sandbox protections. GPU use was
exclusive and sequential. The slot was released after the final smoke.

CPU checks (no Genesis or torch initialization):

```sh
.venv/bin/python -m pytest -q tests/test_sim_backend.py tests/test_sim_mesh.py tests/test_coverage.py
.venv/bin/ruff check src/cloth_fling/sim tests/test_sim* tests/test_coverage*
```

Result: **34 passed in 0.52 s**, Ruff passed. Tests exercise topology, winding,
camera axes/pixel centres, first-hit/hidden-layer rejection, bounded picks,
coincident/workspace/separation rejection, projected union with overlap/holes,
negative gains, NPZ/hash/config/shape validation, attached-snapshot rejection,
target/step ownership, velocity preservation, explicit failed reset and cleanup.

The GPU test is explicitly gated; normal pytest skips it unless the following
environment variable is set. Approved host execution command:

```sh
CLOTH_FLING_RUN_GPU=1 PYOPENGL_PLATFORM=egl UV_CACHE_DIR=/tmp/cloth-fling-uv-cache \
  .venv/bin/python -m pytest -q -s tests/test_sim_gpu.py
```

Final result: **1 passed in 17.48 s**, including a reported invalid drop rather
than a falsely accepted preparation. Main measurements:

| Check | Actual result |
| --- | --- |
| Scene build (warm caches) | 4.368 s; 2654 particles, 5106 triangles |
| Flat seed 7 | 0.514 s settling, final RMS speed 0.00001432 m/s; area/reference 0.25/0.25 m2 |
| Attachment | Stable IDs (1979,683); exactly two free flags false |
| 30 next-step endpoint commands, 4 substeps | Maximum error 2.9469e-8 m; instrumented setters touched attached IDs only; no position writes |
| Release velocity | Both particles [0.0500083,0,0.5000010] m/s; bitwise unchanged across release; finite/nonzero free motion afterward |
| NPZ restore | Exact positions/velocities/time and observation ID; zero residual attachments |
| Repeated flat preparation | Maximum position difference 0 m in final run; observation IDs differed (state/image hash remains content-based) |
| Fold seed 11 | Valid; 5.900 s settling, final RMS speed 0.005838 m/s, coverage 0.480375, area 0.12009375 m2 |
| Drop seed 11 | Invalid, 5.413 s wall; Y max 0.6000773 m exceeded the 0.6 m table half-width; attachments cleared, terminal observation rendered |
| Flat RGB cloth chroma min/median/max | 145/146/146 (uint8 max-minus-min) |
| Flat RGB table chroma min/median/max | 0/0/0; table maximum channel brightness 47 |
| RGB foreground `(max>45)&(chroma>20)` | 256/256 agreement with first-surface ray hits on balanced 128 foreground + 128 background pixel samples |
| RGB chroma >30 vs rendered flat cloth depth | Precision=1.0, recall=1.0 on this flat fixture |

The segmentation result is a measured flat-scene compatibility check for Stage 1;
it does not establish masking quality over a reset distribution or dynamic fling.
Depth/geometry were used only by the host-side validation, never by RGB selection.

Retained failure history: initial code incorrectly assumed PBD preserved input
mesh IDs; the 0.005 m particle setting produced 10301 runtime particles. Corrected
to adopt/validate remeshing at declared spacing. A target floor check initially
rejected a legal 6 mm endpoint due to float precision; corrected to particle-radius
floor with 1e-6 m tolerance. Initial fast, off-center 0.6 s drop lift failed; the
reviewed centered smooth 2.5 s lift still failed seeds listed above. Initial tests
also exposed non-bitwise repeated reset IDs; exact replay identity is verified
through snapshot restore instead. GJK mode removes Genesis's small-substep rigid
collision warning; final run retains torch.jit.script deprecation warnings and a
Quadrants template mapper caching warning. These warnings did not prevent the
measured attachment/render/restore checks.

This qualifies the declared local runtime path, flat/fold fixtures, assisted
attachment mechanics, and explicit failure handling. It does not qualify the
random bunch distribution, cloth calibration, grasp contact, laydown/fling
behavior, collection-scale reliability, or any learned policy.
