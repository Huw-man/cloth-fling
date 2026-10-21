# Near-half-fold pilot results: collection and first fit, evaluation incomplete

The pilot prepared all twelve frozen fixtures and trained its first RGB checkpoint,
but stopped during model-assisted collection when the executing worktree disappeared.
The run is terminal and incomplete. Validation, checkpoint selection and held-out
evaluation did not run, so there is no learned-policy success or generalization claim.

## Completed work

| Item | Result |
| --- | --- |
| Frozen fixtures | 12: six training, two validation, four held out |
| Exploration | 24 slots: 23 active episodes plus one carried initialization failure |
| Random action attempts | 59 |
| Initial fit | 2,000 steps on 51 eligible settled samples |
| Model-assisted collection | One completed episode and one unfinalized episode; three attempts |
| Total action intents / dataset rows | 62 / 62 |
| Valid execution rows / settled actions | 61 / 52 |
| Second fit / validation / held-out evaluation | Not run |
| Actions reaching settled contained 80% coverage | 0 |

The best settled, contained action reached **73.9267%** during random exploration
(`exploration-fold-46-+10-uniform-00`, action two). Its third action lost coverage;
73.9% was below the frozen 80% stopping threshold. The best terminal episode reached
**70.1167%** after three actions (`exploration-fold-54-+10-uniform-01`). These are
exploration outcomes, not learned-policy results. Forty-one recorded gains were
negative and remain preserved. Invalid/unsettled rows remain in source datasets;
only eligible settled outcomes enter regression.

The unselected first checkpoint is
`runs/near-half-fold-pilot-v1-corrected/checkpoints/initial.pt`, SHA-256
`e4d181676fe8cfe7845f237cdc8ab84c2b47edc18ed0316e55072293116a0731`.
Its metadata identifies the training datasets and settings. No test data was used.

## Terminal source-loss failure

Executed revision `64e0a74` ran from `/tmp/cloth-fling-near-half-fold`. During the
second refresh episode that worktree was removed; it was also absent from Git's
worktree list. The removal's actor/origin was not established. No removal was
initiated by the pilot coordinator in this execution.

Attempt 62 (`refresh-fold-46--10-mixed-01`, second action) saved its pre-action state,
resolved the RGB grasp and acquired two native pins. Loading the frozen trajectory
then failed because the profile file in the removed worktree no longer existed.
The action is retained as invalid/uncertain and is not retried. Its pre-action hash
matches the preceding terminal state; its released terminal positions and velocities
are exactly equal to that pre-action state. No prescribed laydown motion ran in this
failed action, and no full trajectory was captured for it.

All 62 action cleanup certificates, 26 episode cleanup certificates and independent
scene cleanup passed. All particles were released with velocity preserved. There
are 37 verified consecutive-action state links. The 61 available contact records
contain zero attached-particle friction events and zero positive friction work;
the failed attempt has no full contact record. Dataset verification found no
artifact corruption. The report retains the uncertain refresh stage and missing
final episode receipt, rather than fabricating completion.

Peak recorded material edge ratio was **2.93164**; maximum pin tracking error was
**4.20e-8 m**. Material strain remains a separate limitation. The terminal failure
was source loss, not a measured deformation, contact or state-chain regression.

The native runner charged **2,485.418 seconds**, including the original preparation
charge and its final reporting. Saved-geometry replay reconstruction added 41.586
seconds and zero physical actions. Offline closeout timing is recorded separately;
the combined work remains within the six-hour ceiling. All GPU leases are released.

## Initial timer bug and bounded correction

The first root, `runs/near-half-fold-pilot-v1/`, prepared all twelve fixtures but
failed before its first restore/action because an enclosing stage watchdog occupied
the collector's process timer. Its exact error was `collection requires an unused
process real-time timer`. This was a runner bug, not a native restoration mismatch.

The fix keeps collector-owned deadlines during episodes and bounds preparation,
fitting and preloading separately. Real-collector CPU regression tests cover all
collection/evaluation stages. A one-time transfer into the corrected root copied
the exact fixture bytes, consumed the failed initialization's episode slot, and
carried its 19.873 seconds into the same budget. It repeated neither preparation
nor that episode, leaving at most 71 new episodes and 213 actions. The original
root, receipts and report remain terminal and unchanged. The transfer path cannot
recover a physical/uncertain attempt or be used again for this source.

The implementation passed **596 CPU tests** (eight GPU tests deselected), lint and
source identity checks. Both implementation commits were audited and integrated
locally. The exact executed source is archived with the corrected run. Future
native worktrees must be locked against removal until all run and rendering work
ends; the evidence-closeout worktree was protected this way.

## Review files

All paths below are relative to the project root. No viewer panes were opened.

- Protocol: `docs/near-half-fold-pilot.md`.
- Dataset roots: `runs/near-half-fold-pilot-v1-corrected/episodes/*/dataset/`.
- Executed report: `runs/near-half-fold-pilot-v1-corrected/family-report.json`.
- Failure analysis: `runs/near-half-fold-pilot-v1-corrected/failure-analysis.json`.
- Independent accounting: `runs/near-half-fold-pilot-v1-corrected/independent-accounting-verification.json`.
- Final verification: `runs/near-half-fold-pilot-v1-corrected/final-verification.json`.
- Best terminal exploration replay (70.1%): `runs/near-half-fold-pilot-v1-corrected/closeout-evidence/exploration-fold-54-+10-uniform-01/combined.mp4`.
- Failed episode replay: `runs/near-half-fold-pilot-v1-corrected/closeout-evidence/refresh-fold-46--10-mixed-01/combined.mp4`.

Both replay directories also contain separate overhead and oblique videos. All six
videos decoded completely. The failed action is explicitly shown as a terminal-only
saved state, not an invented motion. Its original runner replay attempt failed after
source loss; closeout replays were reconstructed later from saved evidence.

Remaining model-assisted collection, the second fit, validation, selected-checkpoint
testing and the 6/8 provisional success criterion are unmet. Unused budget does not
authorize restarting this terminal physical campaign.
