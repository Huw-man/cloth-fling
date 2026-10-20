# Leading-edge qualification — 2026-10-19

The frozen `reference_leading_v2` trajectory passed development and all three
confirmations from the exact same archived native reference fixture. This qualifies
controlled placement for this favorable grasp and fixture only. No learning,
varied grasps, diverse resets, continuity campaign or dynamic fling was evaluated.
The original reference failure and milestone 3 remain terminal and unchanged.

## Frozen motion

The portable configuration is
[`configs/reference-leading-v2-frozen.json`](../configs/reference-leading-v2-frozen.json).
It is byte-for-byte the configuration frozen before confirmation. Candidate 7 uses
transfer X=.55 m over 1.5 s at z=.95 m, a four-second forward descent to x=.75,z=.31,
and .8 s drag to x=.80,z=.31. The C2 joins have horizontal speed .1 m/s and zero
acceleration. Analytical peak placement TCP speed is .520250 m/s, with peak
acceleration 1.029675 m/s². The descent's minimum horizontal speed is .00625 m/s;
its grasped edge continues forward throughout. Midpoint Y and tautened separation
remain fixed. The endpoint holds for one second, releases at rest and settles
unsupported for five seconds. Each passing action lasted 19.38 simulated seconds.

The action reserved 15.5 simulated seconds after tautening, leaving a maximum
17.3-second tautening allowance within the 35-second action ceiling. The actual
favorable fixture reached its tautening bound sooner. No physics, material,
contact, appearance, mesh, table or favorable-grasp settings were tuned.

## Native acceptance

All four passing actions demonstrated:

- Three seconds of whole-cloth airborne hold, with minimum surface clearance
  .055976–.056259 m.
- Native pin IDs 2488 and 1190 retained through placement. Maximum tracking error
  was 4.137e-8 m, below the 1e-6 m limit.
- 100% of free vertices behind the midpoint X plane plus .02 m at all eleven
  controller samples spanning the final .2 seconds before release.
- Full tabletop footprint containment and normalized projected coverage at least
  .95. Confirmation coverage was 99.232%, 99.155% and 99.205%.
- Final whole-cloth RMS speeds .000127–.000483 m/s, with the required final five
  consecutive controller samples at or below .01 m/s.
- Native TCP release speed zero, release-velocity error zero, and independently
  persisted native-free cleanup certification.

The resting-slide contact coupon and exact state/RGB restoration were rechecked
before motion. Every attempt restored fixture state hash
`f8016757a781d40b15e686caaecd1029d987c708a9ac8bdbf1a28c22917cf858`. No reset generation or grasp changes occurred.

## Development and confirmation record

Attempt and candidate indexes start at zero. The first complete pass was candidate
7 after eight development attempts. Three checks followed without further tuning.
Every earlier failure and its signed coverage change remains in the native bundles.

| Attempt | Stage | Transfer X (m) | Requested descent (s) | Coverage | Minimum final-window trailing fraction | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | development | 0.45 | 2 | 52.338% | — | far-edge crossing |
| 1 | development | 0.45 | 3 | 46.417% | — | far-edge crossing |
| 2 | development | 0.45 | 4 | 78.982% | 100.0% | coverage below .95 |
| 3 | development | 0.45 | 1.5 | 37.437% | 99.3% | settling timeout |
| 4 | development | 0.45 | 1 | 81.532% | 83.8% | coverage below .95 |
| 5 | development | 0.55 | 2 | 72.130% | 97.7% | coverage below .95 |
| 6 | development | 0.55 | 3 | 56.470% | 60.7% | settling timeout |
| 7 | development | 0.55 | 4 | 98.660% | 100.0% | all gates passed |
| 8 | confirmation | 0.55 | 4 | 99.232% | 100.0% | all gates passed |
| 9 | confirmation | 0.55 | 4 | 99.155% | 100.0% | all gates passed |
| 10 | confirmation | 0.55 | 4 | 99.205% | 100.0% | all gates passed |

Coverage-only failures completed native motion but did not qualify placement;
execution validity and placement acceptance remain distinct. Early edge failures
were released immediately, without claiming successful near-rest release or settling.

## Material strain, reported separately

Placement acceptance is not material-fidelity qualification. The maximum local
native edge-length ratio relative to the rest mesh was:

| Attempt | Peak edge-length ratio | Local edge extension |
| --- | --- | --- |
| 7 (development) | 2.022438 | 102.24% |
| 8 (confirmation) | 1.484293 | 48.43% |
| 9 (confirmation) | 1.405466 | 40.55% |
| 10 (confirmation) | 1.393394 | 39.34% |

These large local extensions must remain visible when interpreting the successful
coverage result. The fixed gross-instability brake stayed at edge ratio 3.0.

## Evidence, replay and budget

Ignored local evidence is retained in `runs/reference-leading-v2/`: all eleven
outcomes, time-zero/boundary/failure/terminal native states, TCP target coefficients
and samples, measured velocities, pin ownership, leading ordering, edge crossing
history, coverage, strain, contact accounting, cleanup certificates, source copies,
source hashes, original execution log, configuration, frozen trajectory, durable
intent ledger and `offline-audit.json`. The audit independently recomputed native
snapshot hashes, final projected coverage, containment and RMS speed, checked every
cleanup certificate and verified exact fixture restoration. All eleven passed this
artifact audit. An artifact audit does not convert failed placements into passes.

Native scene preparation, contact and motion consumed 315.136
charged wall seconds (whole campaign elapsed 315.281 s), eight
development attempts and three confirmations. These are within the authorized
20+3 action and two-hour limits. Each action stayed below 35 simulated/180 wall
seconds. Rendering is separately bounded and accounted in
`runs/reference-leading-v2-replay/replay-budget.json`; rendering consumed
105.861 seconds, for a conservative combined total of
421.143 seconds (about seven minutes). Both stages hold the exclusive RTX 5080 lease while using a scene.

Wide overhead, oblique and combined videos are under
`runs/reference-leading-v2-replay/`. `confirmed.mp4` contains the development pass
and its three confirmations; `failures.mp4` preserves the seven unsuccessful
attempts. `index.html` opens the confirmed clip for review. Replays import the exact
saved native geometry at 10 Hz with zero-order hold and never step physics. They
are reconstructed evaluation artifacts, not original execution capture.

The executed Python source is archived independently of later repository edits.
Post-run offline review added rejection of unused grid candidates 16 and 17,
whose quintics would reverse X direction, and moved the campaign claim outside
the historical fixture directory. Neither candidate was executed. The frozen
candidate's complete polynomial schedule is unchanged, checked byte-for-byte
against the frozen values, and no action was rerun. The claim migration retained
a durable guard throughout. `offline-schedule-audit.json` and
`claim-migration.json` retain these checks. Repeated terminal `reference tune`
invocations return offline; uncertain actions cannot resume. Milestone 3
prepare/qualify/pilot still return its frozen reset limitation offline.

Final CPU verification passed 446 tests (eight GPU tests deselected). The native
campaign provides the task-specific GPU checks; no extra physical actions ran.
See [the implementation contract](reference-leading.md) for invocation and limits.
