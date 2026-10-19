# cloth-fling

A small robotless cloth unfolding project: overhead RGB, two explicitly assisted
grasp points, deterministic motion primitives, and repeated-action episode datasets.
It is inspired by [FlingBot](https://proceedings.mlr.press/v164/ha22a.html).

The first milestone builds the simulation and collection pipeline. A working pipeline
does not establish that the fling primitive improves coverage or that a policy generalizes.
There are no articulated arms, IK, finger closure, physical pinch qualification, trained
model, cloud services, or published dataset dependency in this milestone.

## Install

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Commands below run from
the repository root. Dependencies are pinned by `uv.lock`.

```bash
uv sync --locked --extra dev
uv run --locked cloth-fling doctor
uv run --locked --extra dev pytest -m 'not gpu'
```

For simulation, use Linux with a CUDA-capable NVIDIA GPU and a driver compatible
with CUDA 12.8. The `sim` extra explicitly includes CUDA 12.8 Torch wheels; Genesis
does not declare Torch itself. The intended development target is one RTX 5080.

```bash
uv sync --locked --extra sim --extra dev
uv run --locked --extra sim cloth-fling doctor --probe
```

If the default uv cache is read-only, set `UV_CACHE_DIR=/tmp/cloth-fling-uv-cache`.
Sandboxed GPU visibility can differ from host visibility. Check `doctor` in the same
execution context that will run simulation; do not interpret package installation
or `nvidia-smi` alone as a successful physics/rendering test.

## First local experiment

Begin with the fixed 50 x 50 cm synthetic cloth in `configs/cloth50.json`.

```bash
uv run --locked --extra sim cloth-fling resets \
  --config configs/cloth50.json --kind fold --seed 11 --output runs/reset-11

uv run --locked --extra sim cloth-fling diagnose \
  --config configs/cloth50.json --reset-dir runs/reset-11 --output runs/diagnostic-11

uv run --locked --extra sim cloth-fling collect \
  --config configs/cloth50.json --reset-dir runs/reset-11 --episodes 1 --max-actions 3 \
  --primitive laydown --output runs/episodes-11

uv run --locked cloth-fling dataset verify runs/episodes-11
```

Use `--primitive fling` to inspect the dynamic candidate. Keep broad collection
behind the primitive diagnostic gate in [the milestone](docs/milestone.md).
CLI reset and collection defaults use `fold`. Random single-point `drop` preparation
is available explicitly, but tested seeds failed table containment; its distribution
is unqualified. Those failures are recorded, with no fallback to a different reset.
The controlled baseline is lift, tauten, diagonal laydown, drag, then release.
The fling candidate retains both grasps through swing, lowering and placement;
it releases near the table. Both remain configurable independently of selection.

Within each episode the collector releases, settles, observes, and chooses again
on the evolved cloth. It resets at episode boundaries or after a recorded invalid
execution. Valid negative coverage outcomes are retained. Invalid execution and
action-count termination are distinct from successful unfolding.

`uniform_valid` is the default selection baseline; `--policy geometry_guided` uses
visible geometry for a separate baseline. No learned policy is included yet.

## Boundaries and artifacts

- Learned-policy input is RGB only. Depth, mesh state, attachment IDs, trajectories,
  and coverage diagnostics are privileged execution/evaluation support.
- Coordinates are world metres with z up, and image pixels `(u, v)` with top-left
  origin, u right, v down. Camera projection and pixel-center convention are saved.
- Local datasets use Parquet action rows, PNG RGB, and NPZ/JSON supporting state.
  Detailed trajectories and geometry stay outside the policy observation.
- `runs/` is ignored. Unfinished runs and debugging output remain local.
- Data readers and verification do not import or initialize Genesis or Torch.
- Future GitHub code, Hugging Face datasets, and Hugging Face models will be separate
  releases with immutable revisions, checksums, cards, and evaluation provenance.

See [runtime qualification](docs/runtime.md), [primitive diagnostics](docs/primitives.md),
and [dataset format](docs/dataset.md) for implementation details and measured limits.

## Development

```bash
uv run --locked --extra dev ruff check src tests
uv run --locked --extra dev pytest -m 'not gpu'
```

GPU tests are opt-in; follow the runtime document for their qualification command.
Use separate reset families/seeds for development and held-out evaluation. Do not
split related episode frames across train and test. Training, model release,
cross-cloth claims, and physical robot adaptation are later milestones.
