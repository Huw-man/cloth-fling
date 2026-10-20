# Requested frictional diagonal demonstration — 2026-10-19

The user requested a fling using the new table friction and laydown trajectory.
This authorized a separate, bounded demonstration; the earlier campaigns remain
terminal. The intended action was the existing retained-grasp lift, three-second
airborne hold, six-second shallow diagonal to X=.78 m / Z=.31 m, adaptive tug of
at most 2 cm, near-rest release and settling. No dynamic-fling trajectory, tuning
or confirmations were added.

**No motion action ran.** The corrected native contact checks found a physical
consistency failure before the action gate opened. Thirteen of fourteen executed
checks passed. The corrected airborne exclusion and attached-particle exclusion
passed, as did below-threshold edge loading at half the normal timestep.

| Above-threshold edge-load check | Normal timestep | Half timestep |
| --- | --- | --- |
| Native timestep | 1 ms | .5 ms |
| Signed load-window COM travel | 56.787 mm | .183 mm |
| Required breakaway travel | ≥5 mm | ≥5 mm |
| Result | passed | failed |

The half-timestep case failed both the breakaway requirement and agreement within
10% of the normal-timestep displacement. Loads, contact coefficients and material
settings were unchanged. This is evidence of timestep sensitivity in the coupled
simulation; the underlying cause has not been isolated. It is not the previous
airborne-fixture setup mistake. No coefficient, gate or motion was adjusted to
work around the failure. The two zero-friction controls were not reached.

Every executed case has saved native positions, velocities, initial/terminal
states, per-particle contact accounting, load definitions and independently
written cleanup certification. The offline audit verified state and artifact
hashes, recomputed drift, Coulomb bounds and nonpositive friction work. No motion,
coverage, deposited-body sticking or tug success is claimed. Material strain is
reported separately in the audit, without a material-fidelity claim.
Peak contact-coupon edge-length ratio was **1.032125**.

The independent one-shot intent, frozen configuration and source copies are under
`runs/reference-diagonal-demo-20261019/`, along with the terminal contact report,
native budget, execution log and audit. Its native allowance is closed and cannot
resume. The failed contact reservation retains its full 60-second charge.
Replays under `runs/reference-diagonal-demo-20261019-replay/` reconstruct saved
contact cases without stepping physics. Their separate render-only receipt
carries the entire native charge into the combined 1,800-second ceiling; it does
not reopen the failed native ledger.
Charged native work was **150.499 seconds** and rendering **65.322 seconds**, for
**215.821 seconds combined**. All three 123-frame MP4 files decoded without errors.

The earlier `runs/reference-diagonal-v3/` artifact hashes remain unchanged. Further
motion is blocked on resolving and rechecking the timestep-dependent contact
response.
