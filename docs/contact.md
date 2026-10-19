# Native table contact, milestone 2

`native_coulomb_v1` is a provisional simulation contact profile for one fixed,
horizontal Box table and one robotless PBD cloth. It is an engineering adaptation
of Genesis 0.4.3, not a measured terry-towel/desk material calibration. Attachment
remains explicit particle assistance, never a physical pinch-grasp claim.

## Contact law and isolation

`sim/native_contact.py` installs a new `LegacyCoupler` subclass instance on this
scene's simulator **before build**. It calls the unchanged upstream normal
function, then changes only tangential velocity. Installed Genesis files and
classes are untouched. The version must be exactly 0.4.3. AST SHA-256 guards pin
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
| `configs/contact-native-v1.json` | .40 | .30 | provisional engineering prior |
| `configs/contact-low-v1.json` | .25 | .20 | sensitivity prior |
| `configs/contact-high-v1.json` | .60 | .45 | sensitivity prior |

Use `SceneConfig(contact_profile="native_coulomb_v1", table_static_friction=.4,
table_kinetic_friction=.3)`. Omitted native table coefficients resolve to .40/.30.
`static_friction` and `kinetic_friction` still configure Genesis cloth/self-contact;
these remain .15/.15 by default. They are distinct from the new table fields.
`table_friction=False` sets native table friction to zero while preserving normal
contact. The profile still identifies the native path.

Historical `SceneConfig` defaults remain `legacy_mg_v1`. That path uses the old
poststep, narrow support-window `mg` adapter. Its default JSON serialization is
unchanged, so old configuration hashes/snapshots remain usable. Explicit new table
coefficients are also available to the legacy adapter. Native scenes skip the
legacy adapter entirely; both cannot apply. New profiles or coefficients change
physics identity and therefore snapshot hashes; restoring across physics profiles
is intentionally rejected. Reset construction and primitive trajectories are
unchanged. Appearance hooks only affect the table renderer and are excluded from
physics config and snapshot identity.

## Diagnostic API and evidence

```python
from cloth_fling.contracts import SceneConfig
from cloth_fling.contact_diagnostics import run_contact_diagnostic

report = run_contact_diagnostic(
    SceneConfig(contact_profile="native_coulomb_v1"),
    output_dir="runs/contact-example", backend="gpu",
    duration_s=.6, include_half_dt=True,
)
```

The config may also be a `SceneConfig` dictionary. The standalone module command is

```bash
PYTHONPATH=src python -m cloth_fling.contact_diagnostics \
  --config configs/contact-native-v1.json --output-dir runs/contact-example
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
- The installed Genesis 0.4.3 `engine/couplers/legacy_coupler.py` normal function
  explicitly omits friction. Its executable source is pinned by the hashes in
  `native_contact.py`; current online [Genesis PBD cloth documentation](https://genesis-world.readthedocs.io/en/v1.4.0/api_reference/engine/material/pbd/cloth.html)
  describes material fields but is a different version and does not establish
  0.4.3 rigid-table friction behavior.

No directly applicable published coefficient for the actual terry towel, desk
finish, moisture, pressure and pull conditions has been established here. The
three coefficient pairs are engineering priors, never literature measurements or
simulated calibration promoted to a real material match. This work qualifies no
learned model, general reset distribution, physical grasp, or deployment.

## Bounded host evidence, 2026-10-19

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
