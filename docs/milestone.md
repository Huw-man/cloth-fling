# First milestone

The approved research plan starts with one 50 x 50 cm cloth and one local NVIDIA GPU.
FlingBot source examined: c5f35c785c6bc5d49282fb10117aa66ca04bf144.
Genesis source inspected during planning: ee6499d92c4f391fb15cb2bac856901b8382e986.

Stage 0: qualify a Genesis cloth/table/two-point scene, visible-surface attachment,
coverage measurement, deterministic reset inputs, save/restore, and release momentum.
Stage 1: implement deterministic lift/tauten/laydown/drag/release and retained-grasp
swing/lower/place/release candidates with measured cloth diagnostics. This milestone
requires a bounded diagnostic command, not a completed 600-execution campaign.
Stage 2: collect repeated actions on evolving cloth and store one Parquet row per
attempt, RGB PNGs, and privileged NPZ/JSON support. Verify ancestry, evidence hashes,
coverage arithmetic, chain continuity, invalid outcomes, and restart behavior.

Acceptance: declared install, offline dataset tools without simulator initialization,
CPU contract tests, host RTX 5080 scene/attachment/restore smoke, and at least a short
repeated-action collection producing a verifiable dataset. Report primitive quality
separately from pipeline function. Do not launch training, broad collection, paid
compute, hardware execution, or publication for this milestone.

Later: locked train/validation/test reset groups, spatial RGB immediate-gain learning,
held-out policy comparison, cloth generalization, HF releases, and hardware adaptation.

Primary references:
- https://proceedings.mlr.press/v164/ha22a/ha22a.pdf
- https://proceedings.mlr.press/v164/ha22a.html
- https://proceedings.mlr.press/v164/ha22a/ha22a-supp.zip
- https://github.com/real-stanford/flingbot
