# Cloth-fling first milestone

Build a standalone Python package, robotless Genesis scene, deterministic assisted
attachment, retained-grasp laydown/fling primitives, bounded repeated-action collection,
and Parquet/image/state dataset tools. Training/model releases and publication are later.

## Coordination

- Work in your assigned Herdr worktree and branch. Commit your owned paths only.
- The coordinator integrates to main. Do not merge, push, publish, or start broad campaigns.
- Shared interfaces are in `cloth_fling.contracts`; request contract changes from coordinator.
- Keep Genesis imports lazy. Offline dataset tools must work without Genesis or torch.
- Use world metres with z up, image (u,v) top-left, u right and v down.
- Never reset between valid actions. Record invalid attempts and terminal next observations.
- Keep negative coverage gains. Do not treat hand paths or regression loss as behavior proof.
- Attached particles only may receive motion commands. Release must retain velocity.
- Privileged geometry/depth/state are execution/evaluation artifacts, never learned RGB input.
- No arms, IK, pinch grasping, AWS, neighboring-checkout imports, or machine-specific paths.

## Runtime

The user explicitly authorizes the host NVIDIA RTX 5080 (16 GB). Sandbox device masking
does not establish a host failure. Use approved host execution for bounded GPU qualification
and short smoke tests. Coordinate GPU use with the coordinator; avoid concurrent scenes.
Do not disable agent sandbox/approval protections. Use `UV_CACHE_DIR=/tmp/cloth-fling-uv-cache`.

## Ownership

- Stage 0 agent: `src/cloth_fling/sim/backend.py`, `sim/mesh.py`, `sim/coverage.py`,
  `tests/test_sim*`, `tests/test_coverage*`, `docs/runtime.md`.
- Stage 1 agent: `src/cloth_fling/sim/primitives.py`, `diagnostics.py`, `selection.py`,
  `tests/test_primitives*`, `tests/test_selection*`, `docs/primitives.md`.
- Stage 2 agent: `src/cloth_fling/data/`, `collection.py`, `tests/test_data*`,
  `tests/test_collection*`, `docs/dataset.md`.
- Coordinator: package configuration, contracts, CLI, integration, lock, README.
