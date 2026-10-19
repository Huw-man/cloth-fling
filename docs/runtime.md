# Genesis runtime and simulation boundary

## Genesis 1.4.4 migration

Package 0.2.0 pins Genesis 1.4.4 and CUDA 12.8 Torch in `uv.lock`. Python 3.11 and
3.12 remain supported; the shared migration target is Python 3.12.13. The exact
[release and dependency record](genesis144-migration.md) separates installation,
offline checks and subsequent runtime/contact/rendering qualification.

Importing the package, CLI, provenance or offline dataset readers does not import or
initialize Genesis/Torch. `cloth-fling doctor` reports installed distribution versions
and GPU visibility without building a scene. The root `.venv` is now upgraded from
the lock. A separate environment under ignored `runs/genesis144/venv` supported the
migration; worktrees using its interpreter must select their own source through
`PYTHONPATH`. Neither environment is a dependency of the installed package.

```bash
export UV_CACHE_DIR=/tmp/cloth-fling-uv-cache
uv sync --locked --extra sim --extra dev --extra video
uv run --locked --extra sim --extra dev --extra video cloth-fling doctor
uv run --locked --extra dev pytest -m 'not gpu'
```

The host target remains NVIDIA RTX 5080, 16 GB. GPU construction requests CUDA explicitly
and rejects silent CPU fallback. Sandbox device masking is not evidence of a host failure.
A compatible driver and working EGL/OpenGL runtime are required for headless rendering.
For sandboxed import checks with a shared read-only environment, set
`NUMBA_CACHE_DIR=/tmp/cloth-fling-numba-cache` and
`MPLCONFIGDIR=/tmp/cloth-fling-matplotlib` so dependency caches are writable.

Only after runtime/contact and rendering code is integrated, CPU readiness is reviewed,
and the coordinator grants the exclusive GPU lease, the bounded scene/render probe is:

```bash
PYOPENGL_PLATFORM=egl uv run --locked --extra sim cloth-fling doctor --probe \
  --config configs/contact-native-v2.json --appearance-config configs/appearance-matte.json
```

This command is a construction/render smoke, not contact or behavior qualification.
Native-v2 contact acceptance must precede any dependent primitive qualification.
No primitive campaigns or collection are authorized by this migration document.

The preferred upgraded CLI profile is `native_coulomb_v2`, with the runtime agent's
`configs/contact-native-v2.json`. Historical v1 profiles keep their original identity.
The 1.4.4 legacy PBD/rigid source still omits tangential friction. The v2 adapter needs
a source-guarded review and fresh evidence; adding 1.4.4 to an old private-API version
whitelist is insufficient. Rigid contact release notes are not cloth friction evidence.

Schema-2 snapshots record the engine version, and physical restore must reject historical
schema-1 / Genesis 0.4.3 state. Dataset reading and `replay --inspect` remain offline and
compatible with historical artifacts. Rendering historical geometry requires an explicit
render-only route, topology validation, source engine identity and separate rendering
capabilities. Never restore old physics silently to make a replay image.

## Historical Genesis 0.4.3 implementation notes

Everything below describes the previous implementation and its measurements. None of
its mesh counts, render-error bounds, restoration checks or contact results establishes
a pass under Genesis 1.4.4. See [milestone 2 historical results](milestone2-results.md).
The current code must be rechecked against the upgraded runtime before simulation resumes.

## Scene and physical assumptions

One 0.50 m square cloth, a 1.20 m fixed table, and two deterministic assisted grasp points.
World units are metres, z up. There are no arms, IK, fingers, pinch, or grasp-contact claims.
The synthetic cloth has areal density 0.30 kg/m2, no air drag, stretch compliance 1e-7,
bending compliance 1e-5, and static/kinetic friction coefficients 0.15/0.15. This material
is not calibrated to a particular physical towel.

Outer physics dt is 2 ms with four substeps. `max_stretch_solver_iterations` is explicit;
the historical configuration used 64. Bounded frozen-state lift checks showed substantial
iteration sensitivity, so Genesis's default four sweeps were inadequate for this slice.
More iterations alone do not establish physical fidelity or convergence of every phase.
Genesis 0.4.3 tied 1 cm remeshing spacing to a 1 cm particle collision diameter
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
