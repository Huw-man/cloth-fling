# Reference single-vertex qualification — 2026-10-19

The reference environment and learning-ready action/data interfaces are implemented.
Bounded host qualification stopped at controlled placement. It does not qualify
repeated-action behavior, diverse bunches, a trained grasp policy or dynamic fling.
Milestone 3's terminal benchmark was neither changed nor resumed.

## Native results

The run used Genesis 1.4.4/CUDA on the RTX 5080, the reference physical profile,
two single-vertex hard pins and a fixed favorable pair near one short edge.

| Check | Observed result |
| --- | --- |
| Native mesh | 2,536 particles; 4,870 triangles |
| Native rest surface area | 0.2398698571 m²; authored target 0.24 m² |
| Exact state/render restoration | Passed |
| Resting-slide contact coupon | Mean horizontal speed 0.1 → 0.00005025 m/s in 0.1 s |
| Entire towel airborne hold | 3.0 s; minimum surface clearance 0.0558774 m |
| Maximum hard-pin tracking error | 3.97 × 10⁻⁸ m |
| Peak measured edge-length ratio | 1.67690 |
| Motion outcome | Stopped at solver boundary during drag, at 9.32 simulated s |
| Terminal tabletop containment | Approximately 34.54% |
| Native release/cleanup | Passed; preserved pinned velocities exactly |
| Continuity episodes | Not evaluated; prerequisite placement did not pass |

During laydown/drag, some free cloth traveled beyond the table's positive-X edge,
fell below the tabletop and reached the solver-domain boundary. The guard stopped
execution. This is a negative placement result; full-cloth lifting alone does not
establish successful unfolding. Cleanup released the pins during the failed action,
so its release is not evidence of the planned near-rest release phase.

The split operator passed the bounded resting-slide check. This does not establish
IPC friction equivalence, broad landing/edge-contact fidelity or material fidelity.
No parameters, contact law, strain thresholds, motion or selector were tuned to
improve this result. No dependent episodes or policy training followed.

## Evidence and budget

Ignored local artifacts are under:

- `runs/reference-laydown-v1/`: initial preparation failure. The renderer's native
  material check caught a missing table texture installation before motion began.
- `runs/reference-laydown-v1-render-fix/`: corrected rendering setup, with all prior
  budget charges carried forward. Includes configuration/ancestry, source files and
  hashes, native fixture, contact terminal state, single-action geometry/velocity
  archives, commands, per-action cleanup certificate, final cleanup and report.

The initial setup failure was charged its full 600-second reservation. The corrected
run used 17.892 s preparation, 0.216 s contact work and 14.498 s diagnostic execution
including artifact persistence inside the action wrapper. The ledger totals about
632.61 charged simulator seconds, one diagnostic action and zero continuity actions.
Artifact/report overhead outside reservations is bounded by the whole-run deadline
checks. The exclusive GPU lease was released.

The executed source is preserved in the corrected run's `source/` directory.
Subsequent software work added terminal-state retention, replay inspection and
formatting; it did not rerun motion or replace these results. The original failed
run lacks a full terminal native-position snapshot; its sampled native trajectory
ends at 9.30 s, while the recorded action stopped at 9.32 s. This gap remains
explicit rather than being reconstructed as executed evidence.

Inspect saved traces offline, without Genesis or Torch initialization:

```bash
cloth-fling replay runs/reference-laydown-v1-render-fix --inspect
```

The new qualification entrypoint refuses existing output directories. `--previous`
can carry charges forward for a terminal setup-only correction; it rejects a
predecessor containing motion attempts. No failed motion is automatically replayed.

## Software verification

The integrated CPU suite passed **432 tests**, with eight GPU tests deselected;
Ruff passed. New checks cover reference geometry/material mapping, action-map
round trips, predictor input boundaries, contact exclusions/dissipation, native
release velocity preservation, cleanup/ownership failures, signed settled-only
learning samples and carried budget limits. These are software checks, not additional
physical qualifications. Offline replay inspection accepted the failed native archive.
The real milestone 3 prepare/qualify/pilot entrypoints returned `pool_limitation`
without importing Genesis/Torch or changing the frozen manifest, budget or lease.
