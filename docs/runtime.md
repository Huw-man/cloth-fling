# Genesis runtime and simulation boundary

The installed simulator is Genesis 0.4.3, pinned with CUDA 12.8 Torch in `uv.lock`.
The planning source revision is recorded in `milestone.md`; packaged APIs were checked
again during implementation. Importing backend/data/policy-support modules does not
initialize Genesis or Torch. GPU construction explicitly requests CUDA and rejects a
silent CPU fallback. Host validation targets the RTX 5080, 16 GB, driver 595.91.07.

## Scene and physical assumptions

One 0.50 m square cloth, a 1.20 m fixed table, and two deterministic assisted grasp points.
World units are metres, z up. There are no arms, IK, fingers, pinch, or grasp-contact claims.
The synthetic cloth has areal density 0.30 kg/m2, no air drag, stretch compliance 1e-7,
bending compliance 1e-5, and static/kinetic friction coefficients 0.15/0.15. This material
is not calibrated to a particular physical towel.

Outer physics dt is 2 ms with four substeps. `max_stretch_solver_iterations` is explicit;
the current configuration uses 64. Bounded frozen-state lift checks showed substantial
iteration sensitivity, so Genesis's default four sweeps were inadequate for this slice.
More iterations alone do not establish physical fidelity or convergence of every phase.
Genesis currently ties 1 cm remeshing spacing to a 1 cm particle collision diameter
(5 mm support radius). This is a coarse contact thickness, not a measured fabric
thickness. Near-table placement uses 20 mm; the former 8 mm target could squeeze a
held layer into the lower layer's collision envelope. The hard 10% strain guard and
0.01 m/s settling criterion were preserved during the correction.

**Table contact is an explicit adaptation.** Genesis 0.4.3's legacy PBD/rigid coupling
resolves normal contact without tangential friction. `sim/contact.py` adds a Coulomb
velocity projection after each outer step, controlled by `table_friction` in SceneConfig.
It applies only to free particles at the finite table's support height that are not
separating. Sliding speed decreases by at most mu_k * g * dt; static friction arrests
small tangential speed. It changes no positions, normal velocities, airborne particles,
attached particles, or particles beyond the table. Tests check dissipation, localization,
dt scaling and zero-friction identity. This uses a resting-load mg approximation, not
measured normal impact impulses or load redistribution through stacked cloth layers.
It must be evaluated independently before a hardware-transfer claim.

The backend alone owns stepping. Motion commands affect only attached particles;
the separately declared contact physics may dissipate supported free-particle velocity.
Release removes attachment without zeroing momentum. No global damping was tuned.

## Geometry, grasps and observations

Genesis isotropically remeshes PBD cloth. In the tested environment the requested 51x51
input grid becomes 2654 particles and 5106 triangles. Runtime ordering, faces, rest edges
and hashes are preserved rather than assuming input OBJ vertex IDs survive loading.

Grasp pixels ray-test the first cloth surface. Eligible particles are visible vertices
of that triangle within 1.5 mesh spacings, ordered by distance then particle ID. Hidden
layers, coincident IDs, misses, unsafe separation and workspace violations reject before
attachment. Selection erodes RGB foreground by a full 3x3 neighborhood and reserves a
mesh-snap margin. Geometry is privileged execution/baseline support, never learned input.

The overhead camera is perspective at (0,0,1.8). Calibration saves intrinsics K,
camera_to_world with OpenCV axes, image_size, pixel_center_offset=0.5, positive optical-z
depth in metres, and world_up=+z. Pixel [v,u] unprojects with
`depth[v,u] * inv(K) @ [u+0.5,v+0.5,1]`, then transforms into world coordinates.

Genesis's first offscreen draw can miss dynamic geometry uploads, and same-step restores
can hit rendering caches. The adapter warms graphics buffers once and forces visual
refresh without stepping physics. The nonflat regression compares RGB/depth against
physics triangles, checks immediate restore and verifies the physics snapshot is unchanged.
Rendering compatibility details are version-pinned, not a general Genesis checkpoint API.

The original visual mesh also interpolated across folds, and native depth rendering
culled folded backfaces. `sim/render_mesh.py` now installs a detached copy of the exact
physics topology with identity visual supports. A camera-local two-sided depth-only
pass supplies pixel-center optical depth. This changes rendering fields only; tests
verify physics state is unchanged. Frozen folded/raised fixtures measured approximately
5 micrometres maximum sampled depth error after this correction.

Coverage is the union of projected physics triangles on fixed 2.5 mm metric cells over
the table. Both observation and flat-reference area use that grid. Overlaps count once,
concavities remain, off-table area is clipped, and coverage/gain are not clamped.

## Reset and restore

`flat` is a controlled fixture. `fold` uses a continuous semicircular crease of radius
twice mesh spacing, then a seeded planar rotation and bounded settling. The prior sharp
15 mm height discontinuity overstretched crease edges by 50-163%; it was removed.
The continuous initialization has negligible extension and approximately 1.1% maximum
chord compression on the tested runtime mesh.

`drop` samples a seeded material point, lifts it smoothly over 2.5 s to table-center XY
at height 0.5-0.9 m, holds 0.5 s, releases, and settles. Its diverse-bunch distribution
is unqualified. CLI reset/collection defaults explicitly use fold; library callers can
request a named preparation. No failed drop silently becomes a fold or flat cloth.

Settling requires RMS vector speed below 0.01 m/s continuously for 0.5 s within ten
simulated seconds. Reset checks finite state, table containment and support height.
Failure records preparation metadata and clears attachments. Collection stores failed
preparations as zero-action episodes, separately from attempted-action rows.

Snapshots store positions, velocities, free flags, runtime topology/rest geometry,
config/calibration identity, simulated time, reset metadata and a content hash. Restore
validates before writes, rejects attached snapshots and mismatched configurations, clears
current attachments, and verifies exact readback. No pickle is required. GPU preparation
may differ at floating point precision; replay uses saved actual state, not seed equality.

## Validation

```bash
uv run --locked --extra dev pytest -m 'not gpu'
CLOTH_FLING_RUN_GPU=1 PYOPENGL_PLATFORM=egl \
  uv run --locked --extra sim --extra dev pytest -q -s tests/test_sim_gpu.py
```

GPU use must be serial/exclusive. Sandbox device masking differs from host visibility.
See `milestone1-results.md` for integrated measurements, artifact locations, qualification
limits and remaining primitive/reset risks. Runtime and dataset integrity are distinct
from demonstrated policy improvement or physical-grasp qualification.
