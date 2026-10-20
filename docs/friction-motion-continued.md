# Stable-motion continuation demonstration — 2026-10-19

The user removed the deposited-cloth drift requirement as an exploratory stop:
continue the simulation while physical execution remains stable. The new explicit
`demonstration` mode records placement-quality violations and continues through
the bounded motion. Qualification mode keeps its historical behavior, and neither
old campaign nor either previous motion attempt was resumed or rewritten.

**One complete native motion ran**, using the same table fixture, favorable pins,
updated position-friction physics, 1 ms timestep and six-second diagonal trajectory.
The action completed in **20.08 simulated seconds**, including attachment, lift,
tautening, three-second airborne hold, diagonal laydown, stabilization, endpoint
hold, release and five seconds of unsupported settling. The tug controller found
its geometric slack target already satisfied and commanded **zero extra pull**.

| Measurement | Result |
| --- | --- |
| Final normalized projected coverage | .979767 |
| Final tabletop containment | 100% of vertices |
| Final whole-cloth RMS speed | .00005389 m/s |
| Release TCP speed | 0 m/s |
| Maximum pin tracking error | .0411 µm |
| Whole-cloth airborne hold | 3 s, minimum clearance 56.06 mm |
| Maximum deposited-cohort p95 drift | 39.77 mm |
| Fraction deposited by grasp height .40 m | .140884 |

The 10 mm drift and 25% deposition criteria were observed violations rather than
abort conditions. Their first violation times are archived, and their strict
placement assessments remain false. They did not prevent a finite simulation,
complete release or settled final cloth. The review replay labels the exploratory
motion's completion separately from the strict placement assessments.

Native contact accounting stayed within its Coulomb impulse bound, applied no
friction to attached vertices, and reported zero positive friction work (total
friction work −.34518 J). State remained finite; ownership and native cleanup were
verified, including unchanged release velocities. The offline audit checked native
snapshot hashes, exact parent geometry, identical physical runtime identity and
unchanged trajectory/configuration against the previous interrupted attempt.

Peak local material edge-length ratio was **1.736297**. This substantial local
strain is reported separately; material fidelity is not qualified. The known
half-timestep contact discrepancy is unchanged. Completion at the normal timestep
is exploratory evidence, not general contact or motion qualification.

Preparation and native motion were charged **35.639 wall seconds**. Rendering uses
a separate receipt carrying that charge into a combined 1,800-second ceiling.
The original and review renders took 12.829 and 12.876 seconds, respectively:
**61.343 seconds combined** with native execution. Both render sets are preserved;
the three final review MP4 files decode without errors (303 frames, 20.2 seconds).
All 480 CPU tests passed, with eight GPU tests deselected; the focused checks cover
continuing past cloth slip, deposition failure and incomplete settling while
retaining numerical/domain/deformation/ownership/deadline stops. Native execution
and independent saved-artifact auditing provide the actual simulation evidence.

The sealed source, configuration, fixture, commands, per-particle contacts, native
boundary/termination states, cleanup certificates, CPU log, ledger and audit are
under `runs/friction-motion-demo-continued-20261019/`. The replay contains this one
completed motion and no contact-test clips:

`runs/friction-motion-demo-continued-20261019-review/combined.mp4`

Overhead and oblique files are available in the same directory. Replay reconstructs
saved native states without stepping physics. No viewer pane was opened.
