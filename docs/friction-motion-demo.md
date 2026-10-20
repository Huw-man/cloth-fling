# Exploratory table-start motion with updated friction — 2026-10-19

After the timestep-dependent contact failure was disclosed, the user renewed the
request for a motion demonstration. This separate exploratory run uses the
unchanged `reference_position_coulomb_v1` physics at its normal 1 ms timestep.
It does not retry or reopen either closed qualification campaign, and does not
claim contact or motion qualification. No contact coupons ran in this request.

**One actual native motion ran.** The cloth started from the exact archived table
fixture, under the new physical identity, with the same favorable single-vertex
pins. It attached, lifted to X=.10 m / Z=.95 m, tautened, held airborne for three
seconds, and entered the existing six-second shallow diagonal toward X=.78 m /
Z=.31 m. The planned adaptive tug, endpoint hold and normal release/settling were
not reached.

The action stopped after **10.18 simulated seconds** with
`deposited_body_sliding`: a fixed deposited cohort reached **10.157 mm p95 drift**,
above the existing 10 mm limit. The demonstration used the unmodified execution
function, including its ordinary placement stop and all instability, ownership,
domain, time and cleanup checks. No trajectory, friction coefficient or gate was
tuned after seeing the result.

Both pins tracked within **0.0346 µm**. The three-second airborne hold had
**56.15 mm** minimum whole-cloth clearance. All cloth vertices remained inside
the tabletop footprint at termination, but normalized projected coverage was only
**.4173** and RMS speed was **.1694 m/s**. This is a failed, interrupted laydown,
with no near-rest-release or settled-placement claim. Failure cleanup released
both pins, retained native velocities and independently certified all particles
free.

Peak local material edge-length ratio was **1.528826**. This strain is reported
separately from the placement result; material fidelity is not qualified.

Native preparation and motion were charged **23.047 wall seconds**. A separate
render-only receipt carries that charge into a combined 1,800-second ceiling.
Rendering took **10.316 seconds**, for **33.363 seconds combined**. All three
154-frame MP4 files decoded without errors; the playable duration is 10.267 s.
The native one-shot intent and ledger remain terminal. Sources, trajectory,
parent/child fixtures, native boundary/failure/terminal states, contact accounting,
TCP targets, metrics, cleanup and execution log are preserved under
`runs/friction-motion-demo-20261019/`. The offline audit verified hashes, exact
geometry ancestry, actual moving lift/diagonal traces, pin tracking, three-second
clearance and release/cleanup evidence.

The wide overhead/oblique replay is under
`runs/friction-motion-demo-20261019-replay/combined.mp4`. It reconstructs only this
motion's saved native traces, never steps physics and contains no coupon clips.
It is an exploratory failed-attempt replay, not a successful fling qualification.
No replay browser or viewer pane was opened; the file path is the handoff.
