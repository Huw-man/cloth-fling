# Frictional progressive diagonal laydown

This new, independent campaign corrects table sticking before evaluating a long
forward/downward diagonal and a small slack-taking tug. It preserves the earlier
reference campaigns and milestone 3 as terminal evidence. No learning, varied
grasps, broader collection, cloud or hardware execution is included.

## Physical profile

`reference_diagonal_v3` uses `reference_position_coulomb_v1`. A scene-local,
source-guarded Genesis 1.4.4 coupler retains the native normal projection and applies
Coulomb friction to the same-step tangential displacement. Static contact cancels
that displacement within the normal-load bound; sliding subtracts the bounded
kinetic correction. Velocity receives the consistent correction divided by dt.
Normal response, sides, airborne vertices and attached vertices are unchanged.
There are no permanent anchors, adhesive forces or poststep velocity brakes.
Position corrections are explicit contact physics, never free-cloth motion control.

Effective static/kinetic coefficients both remain .45, the geometric mean of .6
cloth and .3375 table settings. Material, mesh, appearance, table and action solver
settings are preserved. Coefficients are engineering priors, not measured fabric
properties. Material strain stays a separate report.

The new schema-2 fixture imports the exact archived positions, velocities and native
mesh under a new contact/runtime identity, retaining the parent hash and favorable
vertex IDs. This is explicit fixture initialization, not a rewritten old snapshot.
Exact child state/RGB/depth restoration is checked independently. Old pass status
is never inherited.

## Ordered gates and commands

```bash
cloth-fling reference contact-qualify \
  --fixture runs/reference-laydown-v1-render-fix --output runs/reference-diagonal-v3
cloth-fling reference diagonal --output runs/reference-diagonal-v3 \
  --contact-report runs/reference-diagonal-v3/contact-report.json
```

One exclusive fixture claim and one durable ledger cover both stages: at most 16
contact diagnostics, 12 development actions, three confirmations and 7,200 seconds
of simulator wall, including rendering. Contact diagnostics have .5 simulated/60
wall seconds each; actions retain 35 simulated/180 wall seconds. Reservations and
intent are durable before native mutation. Failed/uncertain actions cannot resume.
The contact stage cannot be retried in another output directory. The motion stage
can start exactly once after a complete contact pass, with the unchanged source and
frozen configuration. Old terminal commands keep their offline behavior.

The sixteen frozen coupons cover rest, ±coasting, uniform and ±edge loading below
and above Coulomb breakaway, oblique landing, airborne/attached exclusions, two
half-timestep edge-load checks and two zero-friction controls. Native external
force fields are confined to coupon scenes. Loads of .5 or 1.5 times .45Mg act from
.1 to .3 s; the initial .1 s allows flat-coupon relaxation. Rest drift is also
measured after this initial relaxation. Edge loads act on the outer 10% strip and
are normalized by actual native mass participation. Subthreshold unloaded-body
95th-percentile drift must remain ≤1 mm. Above-threshold displacement must exceed
5 mm. Uniform acceleration/deceleration must match Coulomb predictions within
20%; half-dt displacement must agree within 10% (or .1 mm near rest). Any failed
contact gate stops all motion.

## Motion and acceptance

Retain attach .2 s, lift 2 s to midpoint x=.10,z=.95, bounded tautening and three
seconds airborne hold. The twelve frozen candidates are endpoint X .78/.76/.74,
each with diagonal durations 6/8/10/4 seconds in that order. One straight quintic
uses the entire forward travel during descent to z=.31. Midpoint Y and grasp span
remain fixed, and native substeps evaluate the polynomial directly. Analytical
speed/acceleration limits are .8 m/s and 2 m/s². There is no high forward transfer.

Free vertices at least 10 cm behind the pins enter a fixed deposited cohort after
five stable native-contact samples at speed ≤.01 m/s. Their material IDs and initial
landing XY never change. At least 25% of free vertices must have landed by grasp
height .40 m; cohort 95th-percentile drift may not exceed 1 cm during descent.

After at most one second of stabilization, freeze a supported tug-body set of at
least 25% of free vertices. Pull forward at most 2 cm in 1 mm quintic increments
lasting .1 s, each ending at rest. Stop when two longitudinal barycentric material
traces have chord/arclength ≥.98 for five samples, or body median drift reaches
1 mm / p95 reaches 3 mm. The trace ratio is a geometric slack proxy, not a force
measurement. Final tug/endpoint-hold median drift must stay ≤2 mm and p95 ≤5 mm.
Retain one second endpoint holding, velocity-preserving release and five seconds
unsupported settling. Full remaining time is reserved before tautening (cap 20 s).

All prior clearance, 1 µm pin tracking, leading-edge ordering, ≥.95 coverage,
tabletop containment, five final settled samples, near-rest release, release velocity
error and independent cleanup gates remain. Gross edge ratio ≥3 or any ownership,
nonfinite, restoration, rendering, artifact or cleanup regression stops dependent
work. Ordinary placement/slip/settling failures may advance through the frozen grid.
The first complete pass freezes the candidate; three checks follow without tuning.
Budget exhaustion is incomplete, never an extension.

## Evidence

Preserve native time-zero/boundary/failure/terminal states, commands and measured
velocities, normal/tangential corrections, impulses/work, fixed deposited IDs and
anchors, cohort slip, tug stop reason, slack traces, coverage and strain. Cleanup
certificates are durable before reporting. Wide overhead/oblique replays import
saved geometry only and never step physics. Report a failure as a failure, even if
final projected coverage is high. Open the successful replay only if every gate and
all three confirmations pass.
