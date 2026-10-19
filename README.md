# cloth-fling

A small robotless cloth unfolding project: overhead RGB, two explicitly assisted
grasp points, deterministic motion primitives, and repeated-action episode datasets.
It is inspired by [FlingBot](https://proceedings.mlr.press/v164/ha22a.html).

Milestone 2 adds native table-contact qualification, whole-cloth clearance and inspectable
execution to the simulation and collection pipeline. A working pipeline does not establish
that the fling primitive improves coverage or that a policy generalizes. There are no
articulated arms, IK, finger closure, physical pinch qualification, trained models, cloud
services or published dataset dependencies in this milestone.

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
Headless rendering also requires a working EGL/OpenGL driver runtime; `doctor --probe`
checks scene construction and actual RGB/depth rendering in the selected environment.

```bash
uv sync --locked --extra sim --extra video --extra dev
uv run --locked --extra sim cloth-fling doctor --probe \
  --config configs/contact-native-v1.json --appearance-config configs/appearance-matte.json
```

If the default uv cache is read-only, set `UV_CACHE_DIR=/tmp/cloth-fling-uv-cache`.
Sandboxed GPU visibility can differ from host visibility. Check `doctor` in the same
execution context that will run simulation; do not interpret package installation
or `nvidia-smi` alone as a successful physics/rendering test.

## Milestone 2 quickstart

For new experiments, explicitly select `configs/contact-native-v1.json`, matte appearance
and `configs/primitive-default.json`. This retains the 50 x 50 cm synthetic cloth and uses
native table friction bounded by normal contact impulses, with provisional static/kinetic coefficients
0.40/0.30. These are engineering priors, not calibrated terry/desk measurements; see
[contact qualification](docs/contact.md). CLI rendering defaults to matte, and the explicit
appearance JSON below records the intended settings independently of physics.
`SceneConfig()` and `load_config(None)` retain legacy physics for compatibility, so new
runs must supply the native profile explicitly.

Run the bounded contact gate first. A nonzero exit or unresolved normal-energy injection
blocks dependent primitive qualification and collection. Preserve the failed output.

```bash
uv run --locked --extra sim cloth-fling diagnose-contact \
  --config configs/contact-native-v1.json --output runs/m2-contact-native
```

After contact passes, generate a **fresh native-profile reset**, then compare both
primitives from that same saved fixture. Do not restore a milestone 1/legacy snapshot
under changed physics. Cross-profile comparisons require the coordinator's explicit
geometry derivation, parent hash, one settling pass and within-profile exact restoration.
Use fresh output directories for each run.

```bash
uv run --locked --extra sim cloth-fling resets \
  --config configs/contact-native-v1.json --appearance-config configs/appearance-matte.json \
  --kind fold --seed 11 --output runs/m2-reset-11

uv run --locked --extra sim --extra video cloth-fling diagnose \
  --config configs/contact-native-v1.json --appearance-config configs/appearance-matte.json \
  --primitive-config configs/primitive-default.json --reset-dir runs/m2-reset-11 \
  --record-video --output runs/m2-diagnostic-11
```

Both primitives lift initially to 0.30 m, tauten, then raise the grasps further as needed
in bounded 0.05 m increments up to 0.75 m. All cloth surface must clear the table by 0.020 m for 0.100 s
before placement; fling swing and pullback retain that clearance until intentional lowering.
Laydown then places diagonally, drags and releases. Fling retains both grasps through its
swing, pullback and placement, releasing near the table. Motion/settling settings remain
separate from scene physics and appearance; full executed settings stay in evidence.

The single diagnostic above is an inspection example, not the complete acceptance gate.
Before collection, complete the [approved milestone 2 comparisons](docs/milestone2.md):
legacy versus clearance motion for both primitives on the original favorable fixture and
predetermined seeds 17/23, exact restoration within each profile, clearance on every accepted
action, and at least 95% final coverage on the original favorable laydown fixture. A fresh
fold reset with seed 11 does not replace that favorable fixture. Only after these gates pass,
run the bounded two-action evolving-cloth episode:

```bash
uv run --locked --extra sim --extra video cloth-fling collect \
  --config configs/contact-native-v1.json --appearance-config configs/appearance-matte.json \
  --primitive-config configs/primitive-default.json --reset-dir runs/m2-reset-11 \
  --episodes 1 --max-actions 2 --primitive laydown --record-video --output runs/m2-episodes-11

uv run --locked cloth-fling dataset verify runs/m2-episodes-11
```

Within an episode the collector releases, settles, observes and chooses again on the evolved
cloth, without resetting between valid actions. An invalid attempt is recorded and ends that
episode; resets occur at episode boundaries. Valid negative coverage gains are retained.
Dataset verification establishes storage integrity, not physical or behavior qualification.
`uniform_valid` is the default selection baseline; `--policy geometry_guided` selects the
separate privileged geometry baseline. No learned policy or broad collection is included.

## Milestone 1 reproduction

Use these explicit legacy settings to reproduce the first milestone's physics, appearance
and motion configuration with the current package. `configs/cloth50.json` selects the old
resting-load table-friction adapter, `--legacy-appearance` restores the original table,
and `configs/primitive-legacy.json` disables the new clearance phase. Historical evidence
and its limits remain in [milestone 1 results](docs/milestone1-results.md) and
[runtime qualification](docs/runtime.md). These commands do not qualify milestone 2.

```bash
uv run --locked --extra sim cloth-fling resets \
  --config configs/cloth50.json --legacy-appearance \
  --kind fold --seed 11 --output runs/m1-reset-11

uv run --locked --extra sim cloth-fling diagnose \
  --config configs/cloth50.json --legacy-appearance \
  --primitive-config configs/primitive-legacy.json --reset-dir runs/m1-reset-11 \
  --output runs/m1-diagnostic-11

uv run --locked --extra sim cloth-fling collect \
  --config configs/cloth50.json --legacy-appearance \
  --primitive-config configs/primitive-legacy.json --reset-dir runs/m1-reset-11 \
  --episodes 1 --max-actions 3 --primitive laydown --output runs/m1-episodes-11

uv run --locked cloth-fling dataset verify runs/m1-episodes-11
```

CLI reset and collection defaults use `fold`; single-point `drop` preparation is available
explicitly. Reset distributions remain unqualified. Failures are recorded without falling
back to a different reset. Use `--primitive fling` to inspect the retained-grasp candidate.

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
The [first milestone results](docs/milestone1-results.md) record the verified episode,
matched primitive comparison, and remaining qualification work.

## Development

```bash
uv run --locked --extra dev ruff check src tests
uv run --locked --extra dev pytest -m 'not gpu'
```

GPU tests are opt-in; follow the runtime document for their qualification command.
Use separate reset families/seeds for development and held-out evaluation. Do not
split related episode frames across train and test. Training, model release,
cross-cloth claims, and physical robot adaptation are later milestones.

## Milestone 2 inspection

The [approved milestone 2 gates](docs/milestone2.md) add native-contact qualification,
whole-cloth clearance, independent matte table appearance, visible debug grasps and
optional execution videos. CLI simulation commands and `cloth-fling replay` default
to matte. Use `--legacy-appearance` for old appearance reproduction, or
`--appearance-config configs/appearance-matte.json` for render-only settings; physics
still comes exclusively from `--config`. Direct backend callers retain the legacy
appearance default.

```bash
# Inspect saved data offline; no simulator or video packages needed.
uv run --locked cloth-fling replay runs/m2-diagnostic-11 --inspect

# Reconstructed saved-trajectory replay (fresh output outside the source).
uv run --locked --extra sim --extra video cloth-fling replay runs/m2-diagnostic-11 \
  --output runs/m2-diagnostic-11-replay
```

Install the optional `video` extra for MP4 encoding. Add `--record-video` to a bounded
`diagnose` or `collect` command for native captures during execution. Recording errors
invalidate and preserve the attempt; no action is retried. Video files stay outside
the dataset manifest under `output/video/`, with relative filenames and hashes in
reports/evidence. Failed contact gates block dependent qualification; successful
storage or rendering alone is not a physical or behavior pass. See
[recording, provenance and replay](docs/dataset.md#milestone-2-evidence-and-recording)
for interfaces, limitations and comparison rules. Final host GPU evidence belongs to
the coordinator; broad collection, training, cloud, publication and hardware work
remain out of scope.
