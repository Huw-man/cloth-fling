# Leading-edge trajectory development

This separately authorized Genesis 1.4.4 campaign retains the archived reference
fixture, scene configuration, contact operator, material, appearance, native mesh,
favorable vertex IDs and single-vertex pins. The original reference campaign and
milestone 3 remain terminal historical evidence. No training or varied grasps run.

The versioned grid is `configs/reference-leading-v2.json`. Its deterministic order
is transfer X .45/.55/.35/.65, each with descent durations 2/3/4/1.5/1 seconds.
After the original attach, lift, bounded tauten and three-second airborne hold,
three quintics transfer horizontally at z=.95, descend forward to x=.75,z=.31,
and drag to x=.80,z=.31. Internal joins retain positive horizontal velocity and
zero acceleration. All segments share position, velocity and acceleration at joins.
Analytical polynomial extrema determine time scaling to .8 m/s and 2 m/s².
Native steps evaluate the polynomial directly. The final attached pose stops, holds
for one second, releases without changing native velocity and settles unsupported
for five seconds. Midpoint Y and tautened grasp separation are retained.

Before any action, the complete schedule reserves placement, endpoint hold,
release and settling from its 35 simulated seconds. Tautening receives only the
remainder, capped at 20 seconds. The action wrapper includes restoration and
artifact persistence in the 180-second wall reservation.

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/cloth-fling-uv-cache .venv/bin/cloth-fling \
  reference tune --fixture runs/reference-laydown-v1-render-fix \
  --output runs/reference-leading-v2
```

Use the installed `cloth-fling` entrypoint equivalently.
A durable exclusive claim beside the archived fixture prevents a new output
directory from extending or replaying this campaign. Existing outputs return their
terminal report offline; an interrupted report is explicitly uncertain. Intent is
persisted before restoration or native motion. Failed/uncertain reservations are
charged their full allowance. No resume, seeds, reset generation or new grasps exist.
The exclusive RTX 5080 lease spans preparation and all native actions. Simulator
reservations are capped at two hours, twenty development actions and three checks.

Each attempt restores the archived schema-2 state hash and original RGB/observation
identity. The historical favorable action must resolve to the same two native IDs.
A fresh bounded contact coupon precedes actions. Rendering/restoration/contact,
nonfinite state, gross strain, solver-domain, ownership, artifact and cleanup
regressions stop the campaign. An ordinary far-edge crossing below z=.35 stops
that placement and releases the pins; placement, coverage or settling failures may
advance through the frozen grid. The first complete pass is frozen before three
confirmations; ordinary confirmation failures leave qualification incomplete, and
no further tuning follows. Fatal failures stop remaining checks.

Acceptance requires positive three-second whole-cloth airborne clearance, retained
pins with tracking error at most 1 µm, at least 95% of free vertices behind the
midpoint X plane plus 2 cm at every controller sample in the last .2 seconds before
release, normalized coverage at least .95, all vertices within the table footprint,
and five consecutive settled samples at RMS speed at most .01 m/s. Release TCP
speed is measured natively and must be at most .01 m/s; native release velocity
error must be at most 1e-4 m/s. Independent cleanup certificates precede reporting.
Strain is archived separately and does not establish material qualification.

Native snapshots include time zero, phase boundaries, failure and termination.
Bundles retain polynomial TCP targets, measured velocities, pin IDs, contact
accounting, leading fractions, edge crossings, coverage, strain and terminal state.
The offline replay loader retains failed attempts and exact recorded geometry;
reference tuning replays use wide overhead/oblique poses and never step physics.

```bash
cloth-fling replay runs/reference-leading-v2 --inspect
cloth-fling replay runs/reference-leading-v2 --output runs/reference-leading-v2-replay --fps 10
```
