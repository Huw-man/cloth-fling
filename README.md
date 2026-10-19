# cloth-fling

Robotless cloth unfolding with overhead RGB, two explicitly assisted grasp points,
deterministic motion primitives, and local Parquet/image/state datasets. Inspired by
[FlingBot](https://proceedings.mlr.press/v164/ha22a.html).

## Install and check Genesis 1.4.4

Version 0.2.0 pins **Genesis 1.4.4**, the latest stable release checked on 2026-10-19,
and starts a new engine/snapshot compatibility boundary. See the
[migration record](docs/genesis144-migration.md) for upstream identity, wheel hash,
locked versions and validation limits. Historical Genesis 0.4.3 results do not qualify
this runtime. Cloth-motion campaigns and collection remain paused during migration.

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Run from the repository root:

```bash
export UV_CACHE_DIR=/tmp/cloth-fling-uv-cache
uv sync --locked --extra dev
uv run --locked cloth-fling doctor
uv run --locked --extra dev pytest -m 'not gpu'
```

The base install reads historical datasets without Genesis, Torch or video packages:

```bash
uv run --locked cloth-fling dataset verify runs/saved-dataset
uv run --locked cloth-fling replay runs/saved-diagnostic --inspect
```

For simulation dependencies, select the `sim` extra. Torch comes explicitly from the
CUDA 12.8 index for the RTX 5080 target; Genesis does not declare Torch itself.
Linux simulation also needs compatible NVIDIA and EGL/OpenGL drivers.

```bash
uv sync --locked --extra sim --extra dev --extra video
uv run --locked --extra sim --extra dev --extra video cloth-fling doctor
```

During migration, set `UV_PROJECT_ENVIRONMENT` to the coordinator-provided shared
migration environment before syncing or running. Preserve the historical root `.venv`
until integration. `doctor` reads package metadata and GPU visibility without importing
or initializing Genesis/Torch. Installation and GPU visibility alone are not physics
or rendering qualification. See [runtime checks](docs/runtime.md) for the separately
leased, bounded upgraded-runtime probe.

## Migration gates

New CLI scenes use the upgraded `native_coulomb_v2` profile and
`configs/contact-native-v2.json` after the runtime/contact integration is applied.
The v1 contact profiles remain archival Genesis 0.4.3 configurations. Do not use an
old contact pass or old snapshot as qualification under Genesis 1.4.4.

The upgraded legacy PBD coupling still omits tangential cloth/rigid friction; rigid
contact improvements in upstream release notes do not establish cloth-contact fidelity.
The guarded v2 extension needs new contact qualification before any dependent primitive
campaign. Do not relax materials, damping or acceptance gates to obtain a pass.
Dependencies and CPU interface checks precede coordinator-assigned, exclusive RTX 5080
probes. There is no GPU lease implied by installation or by these example commands.

Snapshots move to schema 2 with engine identity. Old physical states must be rejected
for restore. Historical Parquet, RGB, NPZ/JSON and offline replay inspection remain
readable. Historical geometry can be rendered only through an explicit render-only
path with compatible topology; this cannot establish physical continuation.

## Historical results and experimental profiles

[Milestone 1](docs/milestone1-results.md) and
[milestone 2](docs/milestone2-results.md) results are **historical Genesis 0.4.3 evidence**.
The previous native-v1 contact measurements, mesh/depth checks, exact restores and
fixed-fixture laydown results do not transfer to 1.4.4. Their reports and failed outputs
remain intact for offline inspection.

`configs/primitive-full-lift-laydown-v2.json` retains an experimental 0.30 m laydown and
0.06 m drag, compared with the 0.12 m reference. Its old fixed-case outcomes are described
in the [dataset notes](docs/dataset.md#experimental-full-lift-laydown-profile).
It is unqualified on the new engine. Dynamic fling, reset distributions, grasp selection
and repeated-action behavior also require separate evidence; none is certified by the
migration or by successful dataset verification. Old profile names are configuration
history, not a command to reproduce old physics in the new runtime.

## Boundaries and artifacts

- Learned-policy input is RGB only. Depth, geometry, attachments, trajectories and
  coverage are privileged execution/evaluation artifacts.
- Units are world metres with z up; pixels `(u, v)` start top-left, u right and v down.
- Motion commands affect attached particles only. Release retains velocity. Explicit
  contact physics is separate from control.
- Valid evolving-cloth actions have no intervening reset; invalid attempts and terminal
  observations are recorded, and valid negative coverage gains are retained.
- Local datasets store Parquet rows, PNG RGB and NPZ/JSON supporting state. `runs/` is ignored.
- There are no arms, IK, physical pinch qualification or trained models. Further primitive
  campaigns, broad collection, training, hardware, cloud and publication remain out of scope.

See [runtime](docs/runtime.md), [dataset format](docs/dataset.md),
[primitive diagnostics](docs/primitives.md) and [viewer/replay](docs/viewer.md).

## Development

```bash
uv run --locked --extra dev ruff check src tests
uv run --locked --extra dev pytest -m 'not gpu'
```

GPU checks require the coordinator's exclusive lease. Package 0.2.0 is a local version
boundary; this migration creates no tag or published release.
