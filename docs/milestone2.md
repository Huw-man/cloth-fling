# Milestone 2: inspectable grasps, table contact, and complete lift

Approved for implementation on 2026-10-19. Keep the 50 cm square cloth and
assisted attachment. Terry-cloth parameters remain provisional. Actual desk/towel
measurements are a later calibration input, not a prerequisite to this bounded slice.

## Delivery sequence

1. Contact: scene-local, pinned Genesis contact extension applies dissipative
   tangential Coulomb impulses bounded by native normal contact impulse. Retain
   legacy physics. Keep table coefficients independent of cloth self friction.
   Provisional static/kinetic settings are 0.40/0.30; sensitivities 0.25/0.20 and
   0.60/0.45 are engineering priors, not published terry-on-laminate measurements.
2. Visuals: neutral gray matte table, roughness 0.8, metallic 0, subtle achromatic
   texture. Keep cloth unchanged. Two red/blue 10 mm spheres show selected targets
   then measured attached vertices, disappearing on release. Debug geometry must
   never affect policy RGB/depth or physics. Preserve the legacy appearance.
3. Clearance: after initial 0.30 m lift and strain-limited tautening, raise both
   grasps smoothly in 0.05 m increments, capped at 0.75 m. Require all cloth surface
   at least 0.020 m above table for 0.100 s before placement. Protect fling swing
   and pullback until intentional lowering. Preserve speed/acceleration/strain,
   30 s simulated/120 s action wall and 10 s settling bounds. Explicitly invalidate
   failures and release without resetting within an episode.
4. Evidence/integration: supported replay and optional native debug recording,
   contact diagnostic CLI, provenance and backward-compatible offline dataset
   inspection. Coordinator merges reviewed stage commits into local main.

## Gates

Contact friction cannot add tangential energy, affect airborne/attached particles,
change normal response, or double apply with the old adapter. Analytic unit checks
and GPU resting/sliding/landing checks precede primitive runs. Target sliding
deceleration within 15% of mu*g and stopping-distance sensitivity under 10% when
halving timestep. Report contact impulses, friction work, vertical rebound and
energy separately. Unresolved normal-energy injection blocks dependent qualification;
no arbitrary damping or coefficient tuning to conceal it.

After contact passes, compare legacy/clearance motion for both primitives on the
existing favorable fixture plus two predetermined reset seeds. Within each contact
profile, settle once and restore that exact derived fixture for each comparison.
New-profile settling does not establish identical initial states across profiles.
Require clearance for every accepted action and >=95% final coverage on the original
favorable laydown fixture. Finish with a verified two-action evolving-cloth episode
only after these gates pass. Retain negative gains and invalid attempts.

## Artifacts and interpretation

Keep local runs ignored. Record source reset hashes, full contact/primitive and
appearance settings, source revision, lock hash, seed, GPU, and replay/video hashes.
Original observations remain clean. Label videos as native execution captures or
reconstructed saved-trajectory replays. Old datasets/videos remain unchanged.
No broad collection, learning, publication or hardware work belongs to this milestone.

## Preliminary material research

No directly applicable primary measurement for dry terry cloth on a laminate desk
was located. Cotton-on-resin experiments use different fabrics and pressures;
terry textile studies often measure skin contact. Do not substitute either for a
calibration of the actual pair. Provide repeated low-load horizontal pull tests,
reporting dry mass, breakaway/steady force, direction, washing/softener history and
condition; estimate coefficients from F/(mg) with uncertainty.

- Cotton/resin tests: https://patents.google.com/patent/JP2000108248A/en
- Terry/skin study: https://pmc.ncbi.nlm.nih.gov/articles/PMC12804228/
- FlingBot code examined: c5f35c785c6bc5d49282fb10117aa66ca04bf144.
  Its adaptive minimum-cloth-height intent informs the bounded clearance gate;
  its simulation movement increments are not physical velocities.
