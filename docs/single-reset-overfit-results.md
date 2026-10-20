# Single-reset overfitting results

The learned selector did not reach the revised **80% coverage milestone** in seven
dedicated greedy checks. Its best check settled at **.754383 after three actions**,
starting from .48615. A learned selection during mixed collection reached .759567.
One uniform exploration action reached **.837467**, with containment, settlement
and cleanup, before later actions lost the gain. This establishes one-instance
reachability of 80%, not a qualified learned policy or repeatable success.

The campaign stopped incomplete after **262 physical attempts**. An exploration
action crossed the existing deformation safeguard: its terminal maximum local
edge-length ratio was 3.072870 against the unchanged 3.0 bound. No further physical
actions, final fit/check, or confirmations ran. The six-hour and 300-attempt limits
were not extended, and the stopped root cannot be resumed.

## Target amendment and execution

The user changed the primary milestone from .98 to .80 during the final part of
collection, before the seventh check. `primary-milestone-amendment.json` records
that instruction. The already-running process retained its .98 stopping rule;
the .80 assessment credits the first settled, contained qualifying action within
each saved episode. Later recorded losses do not erase that earlier observation.
This is an explicit retrospective assessment of earlier actions, not a claim
that the executed episodes stopped at .80.

New runs default to the separately versioned .80 protocol. Explicit
`--coverage-target .98` retains historical reproduction. Both keep immediate
coverage-gain regression, the exact archived `fold-50-+00` fixture, RGB-only model
input, native single-vertex pins, and the approved v4 motion/physics. Changing the
target does not reopen this campaign or authorize a new one.

## Accounting and learning

| Stage | Episodes with physical attempts | Actions |
| --- | ---: | ---: |
| Initial uniform exploration | 20 | 58 |
| Mixed refresh collection | 68 | 184 |
| Dedicated greedy checks | 7 | 20 |
| Confirmations | 0 | 0 |
| Total | 95 | 262 |

There are 94 finalized episode receipts and one failed, unfinalized episode with
three retained dataset rows. Of the 262 attempts, 261 have valid execution
receipts and one is invalid. Twenty-six finalized episodes failed settlement;
the last episode additionally stopped on the native safeguard. Failed episodes
were not replaced and actions were not retried.

Seven fits used the prescribed 2,000 steps each, growing from 51 to 198 eligible
collection labels. Checks never entered fitting. Additional final-round data was
saved but the planned eighth fit was not run. Low regression loss on recorded actions did
not establish reliable action selection.

Completed collection receipts contain 215 eligible labels, of which the last completed fit
used 198. There were 153 negative measured gains. Nineteen of twenty greedy-check
actions overpredicted their observed gain; mean predicted-minus-measured gain was
.318356. These are diagnostics of the sampled actions, not a generalization result.

| Check | Coverage after successive actions | Settlement failure |
| --- | --- | --- |
| 00 | .4140 → .6733 | Action 2 |
| 01 | .2659 → .3342 → .3115 | No |
| 02 | .5362 → .3890 → .4413 | No |
| 03 | .7133 → .3760 → .3523 | No |
| 04 | .3227 → .3175 → .2665 | Action 3 |
| 05 | .5277 → .5141 → .4363 | No |
| 06 | .3721 → .3649 → .7544 | No |

Repeated identical initial RGB, saved native state and grasp selections produced
different coverage outcomes. For example, one grasp produced .414017 and .530133;
the known .837467 exploratory grasp later produced .713317 in a greedy check.
Exact represented-state restoration passed. The cause of outcome variability is
not established. `repeatability-observation.json`, `repeated-grasp-analysis.json`
and `training-point-audit.json` preserve the observations and known-label checks.

## Native stop and reporting correction

The last action was `refresh-07-07`, action 3, selected uniformly. Its native state
was finite, inside the tabletop footprint and fully released; whole-cloth RMS
speed was .101597 m/s. The deformation bound was exceeded, not the solver domain.
The saved pre-action state exactly matches the preceding action's saved terminal
state. `failure-analysis.json` records the hashes and measured bounds.

The executed code raised after the primitive returned, discarding its full
outcome from the collector. A secondary audit then treated the missing handoff as
a state-chain mismatch and masked the original safeguard error. The fix retains
the returned outcome and original failure, and checks failed-action continuity
against the hashed pre-action snapshot when available. Offline reports now expose
unfinalized dataset rows without inventing a completed receipt or permitting retry.

The failed action's before, attached and terminal native states, intent, dataset
row and independent cleanup certificate remain available. Its full trajectory and
recorded duration fields are unavailable. Its replay therefore shows the available
earlier actions and a terminal-only failed-action frame; missing motion is not
reconstructed. The reporting fix was not used to rerun or requalify this campaign.

Material fidelity remains separate from placement. Available completed-action
traces reached a local edge-length ratio of 2.921477; the failed terminal snapshot
reached 3.072870. The failed action's complete peak is unknown because its full
trace was lost. No material, generalization or hardware qualification is claimed.

## Verification and review files

The executed native revision was `92e71aa`. Source identity remained unchanged
through native execution. Target-versioning and reporting fixes were tested and
integrated afterward. The final implementation passes **575 CPU tests** with eight
GPU tests deselected. The real campaign's failure remains part of runtime evidence.

Independent accounting reconciled 262 durable intents and dataset rows across 95
episodes. All 262 action cleanup certificates, 95 episode cleanup certificates,
and scene cleanup passed. All 167 saved state-chain links matched, including the
last action's separate pre-action snapshot. The 261 available contact records had
zero attached-particle friction events and zero positive friction work; the failed
action's missing full outcome prevents a corresponding complete contact audit.
Peak recorded pin tracking error was 4.226e-8 m and release velocity error was zero.

The original execution and offline report charged 11,342.493 seconds (3.151 hours).
Recorded action durations reached 35.000000000028 simulated seconds (roundoff at
the 35-second bound) and 46.973 wall seconds. The failed action lacks those duration
fields. Supplemental replay generation is render-only, uses zero physical actions,
and retains its own elapsed-time receipt within the original six-hour ceiling.
Final artifact verification finished 12,004.657 seconds (3.335 hours) after the
campaign began, also within that ceiling. See `verification.json` in the run root.

Project-root-relative records:

- Original executed report: `runs/single-reset-overfit-v1/executed-overfit-report.json`.
- Primary .80 assessment: `runs/single-reset-overfit-v1/primary-milestone-report.json`.
- Accounting and limitations: `runs/single-reset-overfit-v1/independent-accounting-verification.json`.
- Failure analysis: `runs/single-reset-overfit-v1/failure-analysis.json`.
- Best greedy replay: `runs/single-reset-overfit-v1/primary-milestone-evidence/check-06-00/combined.mp4`.
- Exploratory .80 hit, including later executed losses: `runs/single-reset-overfit-v1/primary-milestone-evidence/exploration-17/combined.mp4`.
- Available failure replay: `runs/single-reset-overfit-v1/primary-milestone-evidence/refresh-07-07-corrected/combined.mp4`.

Overhead and oblique files accompany each combined replay. No viewer pane is opened.
`primary-replays-v2.json` selects the reviewed videos. Its failure replay corrects
a terminal-only table-height annotation using the saved scene configuration;
the superseded first rendering is preserved. `video-verification-v2.json` records
full decoding checks, and representative frames were visually inspected.
No checkpoint is designated qualified; all seven checkpoints and failed evidence
remain archived.
