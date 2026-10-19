# Native table contact, milestone 2

`native_coulomb_v2` is a provisional simulation contact profile for one fixed,
horizontal Box table and one robotless PBD cloth. It is an engineering adaptation
of Genesis 1.4.4, not a measured terry-towel/desk material calibration. Attachment
remains explicit particle assistance, never a physical pinch-grasp claim.

## Migration status

CPU compatibility tests cover the 1.4.4 source guards, archived-profile rejection,
schema-2 exact restore, cross-engine rejection, render-only latching, attachment
momentum and offline import isolation. **Genesis 1.4.4 GPU contact and rendering
qualification are separate:** fresh 1.4.4 native microtests and the three bounded
contact coupon profiles passed on the RTX 5080 (see the evidence section below).
This contact run does not qualify rendered RGB/depth, physical snapshot replay,
primitives, resets or collection. The 0.4.3 results remain historical evidence
and confer no pass status on v2. Further bounded 1.4.4 probes require a new
coordinator lease and the shared upgraded environment; never use the old venv.

## Contact law and isolation

`sim/native_contact.py` installs a new `LegacyCoupler` subclass instance on this
scene's simulator **before build**. It calls the unchanged upstream normal
function, then changes only tangential velocity. Installed Genesis files and
classes are untouched. The version must be exactly 1.4.4. AST SHA-256 guards pin
both the normal function and its calling kernel; changed executable source fails
clearly. Whitespace and comments may differ. Other Genesis versions require a
new review and profile qualification, not a bypass of the guard.

For mass `m`, native input velocity `v`, native normal result `vN`, and unit normal
`n`, the positive normal correction impulse is

```
Jn = m * max(0, dot(vN - v, n))
vt = vN - dot(vN, n) * n
Jstop = m * norm(vt)
Jt = -m * vt                         if Jstop <= mu_static * Jn
Jt = -min(Jstop, mu_kinetic*Jn)*vt/norm(vt)  otherwise
vout = vN + Jt/m
```

The cap comes from the actual native correction, including impact. There is no
`mg` proxy, damping factor, normal restitution change, height/vz support window,
or position overwrite. Friction requires a free particle, a positive native
normal correction, and a top normal with `nz > .9999`. Side contact retains
Genesis response without this friction extension. Equal/opposite tangential
coupling force is also recorded on the fixed table. Normal position, velocity
and coupling response remain the upstream result. A zero-speed tangent is safe.
Airborne particles and attached particles receive zero tangential impulse.

The fixed-table impulse reduces tangential kinetic energy, cannot reverse its
direction, and leaves the normal component unchanged. That property does not
prove the entire cloth constraint/contact system is energy-stable: separate
unforced drop tests monitor normal and total mechanical energy. Unresolved
normal-energy injection stops diagnostic qualification immediately; this stage
adds no normal damping to conceal it.

## Profiles and historical replay

| Configuration | Static table mu | Kinetic table mu | Status |
| --- | ---: | ---: | --- |
| `configs/contact-native-v2.json` | .40 | .30 | bounded coupon gates passed; engineering prior |
| `configs/contact-low-v2.json` | .25 | .20 | bounded coupon gates passed; sensitivity prior |
| `configs/contact-high-v2.json` | .60 | .45 | bounded coupon gates passed; sensitivity prior |

Use `SceneConfig(contact_profile="native_coulomb_v2", table_static_friction=.4,
table_kinetic_friction=.3)`. Omitted native table coefficients resolve to .40/.30.
`static_friction` and `kinetic_friction` still configure Genesis cloth/self-contact;
these remain .15/.15 by default. They are distinct from the new table fields.
`table_friction=False` sets native table friction to zero while preserving normal
contact. The profile still identifies the native path.

`SceneConfig` keeps `legacy_mg_v1` as its omitted-profile archive decoder so old
JSON hashes and offline dataset tools remain readable. Both `legacy_mg_v1` and
`native_coulomb_v1` are rejected by the 1.4.4 backend before initialization. New
simulation callers must select v2 explicitly. The v1 config files are unchanged.
The legacy poststep helper remains available for offline reference tests; new
scenes never execute it. Table appearance remains excluded from physics identity.

Physical snapshots now use **schema 2**, with `engine="genesis"`,
`engine_version="1.4.4"`, `substeps=4`, and canonical `runtime_identity_json`, in
addition to configuration, topology, positions, velocity, attachment flags and
content hash. `restore` rejects schema-1 physical state explicitly, even when the
rest mesh happens to match. Engine/runtime identity, configuration, calibration,
mesh and content checks all precede device setters. No archive metadata editing
or reset conversion makes an old physical state compatible.

The rendering port may call `backend.restore_recorded_geometry(state)` for schema
1 or 2 archives **only if faces, edges, rest positions and mesh hash match
exactly**. This hook verifies the recorded content hash, writes recorded positions
without stepping, and permanently latches that instance into render-only mode.
It zeros velocities rather than claiming to restore dynamics. `step`, `reset`,
`attach`, physical `restore`, and physical `snapshot` then fail. The hook returns an
observation; `recorded_geometry_identity` records source state/schema/engine and
render engine separately. Different triangulations or particle orders are
unsupported, with no interpolation or hidden stepping. Construct a fresh backend
for physics. `runtime_identity` and `capabilities` provide JSON-safe metadata for
CLI/provenance; neither property asserts qualification.

The backend also forwards optional renderer-owned metadata:
`backend.rendering_identity` reads `PhysicsMeshRendering.identity`, and
`backend.rendering_capabilities` reads `PhysicsMeshRendering.capabilities`.
Both adapter attributes should be JSON-safe dictionaries (properties are fine).
Backend access returns a detached copy. Missing metadata remains `None`/JSON
`null`; runtime version alone never implies a renderer capability or passed
qualification. These visual fields are excluded from physical snapshots.

The 1.4.4 PBD public entity setters/getters, `fix_particles(...,
zero_velocity=False)` and `release_particle` retain their roles. The cloth loader
still remeshes to PBD `particle_size`; `Cloth.rho` still denotes kg/m² and mass is
surface area times rho. The scene explicitly selects `LegacyCouplerOptions` and
uses public `RigidOptions(use_gjk_collision=True)`. No material, mass, reset or
primitive values were retuned: dt remains .002, four substeps, 64 stretch sweeps.
The solver-state `get_state` readback still supplies native free flags.

## Diagnostic API and evidence

```python
from cloth_fling.contracts import SceneConfig
from cloth_fling.contact_diagnostics import run_contact_diagnostic

report = run_contact_diagnostic(
    SceneConfig(contact_profile="native_coulomb_v2"),
    output_dir="runs/contact-example", backend="gpu",
    duration_s=.6, include_half_dt=True,
)
```

The config may also be a `SceneConfig` dictionary. The standalone module command is

```bash
PYTHONPATH=src python -m cloth_fling.contact_diagnostics \
  --config configs/contact-native-v2.json --output-dir runs/contact-example
```

The coordinator can connect this API to the main CLI; CONTACT does not edit it.
Runs are serial, bounded to .3–2 simulated seconds per main scenario, and require
an exclusive GPU lease. The airborne control is limited to .1 s, before impact.
The default tests use the full .5 m, 1 cm mesh coupon. Fixtures set initial flat
positions and uniform velocity directly without invoking or tuning reset shapes
or primitives. Rest, positive/negative slide, horizontal drop from .1 m, oblique
landing, fully attached translation, airborne translation, and half-dt positive
slide are independent diagnostic fixtures. Their discontinuous reinitializations
are not an action-collection episode.

Each scenario writes positions/velocities at every full step and **native device
accounting at every substep** to its NPZ. No device readback is added inside the
contact kernel or per substep. Native fields accumulate impulse magnitudes, kinetic
work, and counts for free contact, attached contact, noncontact, and friction
participation; column names are saved alongside values. Work is the actual kinetic
energy change from the friction operation. Native normal work records the kinetic
change of the unchanged upstream response, not total mechanical-energy gain.
Normal impulses include attached native corrections; friction eligibility excludes
them. Cumulative accounting has no action-duration buffer: per-particle float64 totals
and a 64-bit substep counter remain valid across 30 s actions, resets and repeated
episodes. Ordinary scene reset/restore never clears them. Sample
`backend.take_contact_accounting(reset=False)` on demand, for example once per
primitive captured trace. It returns schema/profile, epoch, substeps and JSON
`totals` with the named impulse/work/participation columns. Each sample reads a
fixed-size per-particle array plus one scalar; there is no automatic tick readback.
`reset=True` returns the previous totals, then clears cumulative accounting and
increments the epoch. It never changes physics or diagnostic capture. Differences
within the same epoch give interval totals; do not subtract across epochs.

Per-substep capture is **disabled by default**. Diagnostic calls explicitly opt in
with `coupler.clear_contact_trace()` (start a fresh capture); call
`stop_contact_trace()` to disable it without clearing data. `read_contact_trace()`
returns columns, values, substep count and truncation. Its capacity is 8192 captured
substeps. Only an explicitly requested diagnostic capture fails qualification on
overflow. Cumulative totals continue independently even if capture fills, is
stopped, or is never enabled. The friction calculation never reads accounting or
capture state. Both forms are privileged execution artifacts and never learned
RGB input.

`report.json` records runtime/configuration, full-step energy/rebound/landing
measurements, native impulse/work totals, and separate gates. The native-kernel
microtest independently compares the subclass and original function on identical
SDF inputs, checking exact position invariance and normal-velocity invariance within 1e-6 m/s
(separate inlined float32 native calls can round differently), analytic static and
kinetic impulses, resting `mu*g*dt`, reversed impact, and zero attached/airborne
friction. It also checks friction-only kinetic energy nonincrease.

The unforced coupon normal-energy gate allows at most `max(.001 J/kg, 2% of initial
normal energy)` numerical gain; this tolerance is recorded. Energies are equal
particle averages of kinetic plus gravitational energy relative to the table
support radius. They omit elastic constraint energy and are intended for these
initially unstressed planar coupons, not general folding-energy audits. Full-step
landing samples can miss within-step extrema; native work/impulse accounting covers
all substeps. Mean height determines first supported frame; per-particle state is
retained to inspect outliers. Sliding deceleration is fitted while supported and
above .05 m/s, excluding the first .01 s. Required gates are <15% error from
`mu_kinetic*g`, reverse symmetry, <10% stopping-distance change at half dt,
supported rest, unchanged airborne/attached lateral motion, complete native
accounting, zero excluded impulses, and nonpositive native friction work. A
passed gate qualifies these bounded simulator fixtures only.

## Manual actual-desk/towel protocol (proposed, not executed)

1. Identify the actual towel and desk finish; photograph the contact surface and
   record towel mass/dimensions, weave/pile direction, wash history, dry/moist state,
   temperature/humidity, cleanliness, and any coating or wear. Use the same physical
   pair for all trials. Visual appearance alone does not identify friction.
2. Lay the towel or a documented flat coupon without tension or wrinkles. Record
   the contacting area and any added distributed load. Use a calibrated low-force
   gauge with resolution suitable for the expected sub-newton forces; check zero
   and known loads. Spread the pull across an edge so it translates instead of
   peeling, bunching, or locally stretching. Record video and force/time together.
3. Pull horizontally at table height. Increase force slowly from rest; record the
   peak immediately before sustained slip. Then translate at approximately constant
   speeds (for example .01, .05, .10 m/s); use the steady interval to estimate
   kinetic force, excluding acceleration and stick-slip transients. Record actual
   speed and acceleration; do not confuse `Fpull - m*a` with `Fpull` when accelerating.
4. Repeat at least five times in each of two weave directions and both polarities,
   repositioning onto a fresh documented area. Repeat with a modest distributed
   load to assess pressure dependence. Reject runs that peel, wrinkle, yaw strongly,
   or drag outside the recorded desk area. Retain rejected runs and reasons.
5. For a horizontal pull on an unloaded towel, estimate `mu_s=Fpeak/(m*g)` and
   `mu_k=Fsteady/(m*g)`. Include distributed load in the normal force. If the pull
   angle is nonzero, measure it and correct the normal load to
   `m*g - Fpull*sin(angle)` and the lateral component to `Fpull*cos(angle)`;
   preferably redo at zero angle. Report force/gauge/mass/angle uncertainty and the
   distribution across trials rather than a single fitted constant.
6. Compare those distributions with the sensitivity priors. If friction depends
   strongly on orientation, speed, pressure, humidity, or wrinkle state, state that
   a single isotropic Coulomb pair is insufficient. Validate candidate simulation
   coefficients against held-out physical slide/stop trials before making any
   material-match claim. This document authorizes no hardware operation.

## Sources and limits

- Macklin et al., [Unified Particle Physics for Real-Time Applications](https://matthias-research.github.io/pages/publications/flex.pdf),
  section 6.1, motivates static/kinetic friction bounded by contact correction in
  PBD. This implementation uses the native **velocity correction impulse** rather
  than reproducing that paper's position-friction algorithm.
- [Physics-Based Simulation: Collision Constraints](https://phys-sim-book.github.io/position-based-simulations/other-common-constraints/pbd_collisions_constraints.html)
  describes static and kinetic tangential correction bounds. It supports the
  contact-law design, not values for a particular towel and desk.
- Genesis 1.4.4, upstream `a649105`,
  [`engine/couplers/legacy_coupler.py`](https://github.com/Genesis-Embodied-AI/Genesis/blob/a649105/genesis/engine/couplers/legacy_coupler.py):
  `_func_pbd_collide_with_rigid_geom` explicitly omits friction. New executable
  AST hashes pin that normal function and `kernel_pbd_rigid_collide`. The v2 port
  uses `array_class.RigidInfo` and the new coupling-force argument order
  `(link_idx, batch_idx, pos, force, links_state)`.
- The same wheel's `sap_coupler.py` builds rigid/FEM contact handlers; its
  `ipc_coupler/coupler.py` imports rigid/FEM entities and routes cloth through the
  FEM path. Neither implements the PBD cloth-table friction required here.
  Switching to FEM/IPC would change the physical model and is outside this port.
- `engine/entities/pbd_entity.py` (`PBD2DEntity.sample`) and
  `engine/materials/PBD/cloth.py` establish remeshing and areal-density semantics.
  Review used the verified 1.4.4 wheel source; no installed files were changed.

No directly applicable published coefficient for the actual terry towel, desk
finish, moisture, pressure and pull conditions has been established here. The
three coefficient pairs are engineering priors, never literature measurements or
simulated calibration promoted to a real material match. This work qualifies no
learned model, general reset distribution, physical grasp, or deployment.

## Fresh Genesis 1.4.4 contact evidence, 2026-10-19

Under the coordinator's exclusive RTX 5080 lease, the native impulse microtest
passed and all three profiles passed all 15 existing coupon gates. This is fresh
1.4.4 execution evidence, independent of the historical section below. Execution
used integrated base `fa4bbac` plus the test-only accessor fix committed with this
record: two stale `rigid._rigid_info` references became public `rigid.rigid_info`.
The first microtest attempt failed at that accessor before impulse assertions;
its log is retained. The second attempt passed. No contact-law, normal response,
source guard, material, damping, trajectory, or acceptance threshold changed.

The host was NVIDIA GeForce RTX 5080, 16303 MiB, driver 595.91.07. The separate
shared environment supplied Python 3.12.13, Genesis 1.4.4, Quadrants 1.3.3,
Torch 2.11.0+cu128, and Trimesh 4.12.2. Each host command used this worktree's
`src` through `PYTHONPATH`. The historical root environment was not used.
Native runtime identity recorded CUDA, ndarray mode, deterministic algorithms,
32-bit precision, `.002` dt, four substeps and 64 stretch sweeps. The contact
diagnostic used the unchanged .5 m cloth, 1 cm particle diameter, .3 kg/m² density,
and .6 s main cases; airborne translation remained bounded to .1 s. The slide
comparison used `.001` dt and four substeps. Profiles ran serially.

| Table static/kinetic | Slide deceleration (m/s²) | Expected mu*g | Half-dt stopping-distance change |
| --- | ---: | ---: | ---: |
| .40/.30 | 2.945139 | 2.9430 | .28036% |
| .25/.20 | 1.963931 | 1.9620 | 1.81838% |
| .60/.45 | 4.410652 | 4.4145 | .26139% |

Both unforced drop cases had zero measured positive normal-energy gain in all
profiles. Their maximum sampled mean rebound height was 2.33e-9 m. All cases
had zero recorded positive friction work. Airborne normal/tangential impulses
and attached tangential impulses were exactly zero. The prescribed attached
coupon retained a small native vertical numerical residual: mean height rose
about 4.95e-6 m, with normal-energy gain about 4.86e-5 J/kg, below the unchanged
.001 J/kg gate. These samples do not establish real material restitution or
calibration. The native microtest separately verified normal position/velocity
invariance, analytic static/kinetic/impact friction, reversed motion, excluded
particles, logging-on/off equality, and preservation of cumulative counters
through `scene.reset()` and explicit accounting-epoch resets.

Ignored evidence is retained at
`runs/contact-native-v2/gpu-qualification-20261019/`:

- `native-attempt1/report.json`, `low-attempt1/report.json`, and
  `high-attempt1/report.json`, with eight full-step/substep NPZ archives each.
- `microtest-attempt2/test_native_impulse_microtests0/native-microtests.npz`
  and the passing log; `microtest-attempt1.log` preserves the failed accessor attempt.
- `execution-identity.json`, `executed-test-fix.patch`, `qualification-summary.json`,
  `audit_evidence.py`, and `evidence-audit.log` record source/lock/artifact hashes,
  runtime versions and an independent offline completeness audit.
- `offline-tests.log`: integrated offline suite **322 passed, 3 GPU deselected**.
  Ruff and `git diff --check` passed for the owned change.

The offline audit verified finite state arrays and exact native accounting counts
for all 24 case archives: 1200 substeps per .6 s case, 2400 for half-dt slides,
and 200 for airborne controls. No captured substeps were truncated. The contact
diagnostics constructed the integrated backend and its rendering adapter but did
not render or assess camera images. No primitives, collection, training, cloud,
hardware or publication ran. Runtime metadata does not self-certify arbitrary
configurations; bounded qualification remains attached to these reports.
The GPU compute-app list was empty at handoff and the runtime/contact lease was
explicitly released before this CPU-only documentation work.

## Historical Genesis 0.4.3 evidence, 2026-10-19

The RTX 5080 run used Genesis 0.4.3, .5 m cloth, 1 cm particle diameter,
`sim_dt=.002`, four substeps and the half-dt comparison. All three profiles passed
all reported gates. The offline suite passed 192 tests; the opt-in GPU kernel
microtest passed, including logging-on/off output equality, cumulative reset/epoch
semantics and preservation through `scene.reset()`.

| Table static/kinetic | Measured slide deceleration (m/s²) | Expected mu*g | Half-dt stopping-distance change |
| --- | ---: | ---: | ---: |
| .40/.30 | 2.94540 | 2.9430 | .2405% |
| .25/.20 | 1.96365 | 1.9620 | 1.9343% |
| .60/.45 | 4.41055 | 4.4145 | .2523% |

Horizontal and oblique drop normal energy never exceeded their initial value in
these runs. Maximum sampled mean rebound heights were below 2e-9 m (numerical
projection residual, not a physical restitution measurement). Native positive
friction-work totals were zero. Airborne normal/tangential impulses and attached
tangential impulses were exactly zero in the accounting.

Ignored evidence is retained under `runs/contact-native-v1/`:

- `verified-native/report.json`, `verified-low/report.json`, and
  `verified-high/report.json`, with each scenario's NPZ state and substep accounting.
- `verified-microtests/test_native_impulse_microtests0/native-microtests.npz`.
- `verified-qualification.log` and `cpu-tests.log`.

Earlier build/debug attempts are retained separately in that directory and are
not the final qualification reports. A cancellation-sensitive work calculation
was replaced with the equivalent stable impulse-work identity
`m*dv.dot(v_before) + .5*m*dv.dot(dv)`; no normal response or friction coefficient
was tuned to obtain the result. Full 30 s primitives, reset distributions and
repeated-action behavior were not qualified by CONTACT. Cumulative storage has no
action-duration cap, but this stage's physical evidence is the bounded coupon and
native-kernel suite. Optional appearance integration requires the coordinator's
separate `sim/appearance.py`; CONTACT ran with historical appearance (`None`).
