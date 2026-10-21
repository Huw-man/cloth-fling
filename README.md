# cloth-fling

A small **dataset → RGB grasp-value model → simulation** workflow. Choose two image
grasp points; the environment applies assisted vertex pins and executes a frozen
lift/laydown with a centered 12 cm drag. Inspired by
[FlingBot](https://proceedings.mlr.press/v164/ha22a.html).

Code: MIT. [Dataset v1](https://huggingface.co/datasets/humannew/cloth-fling):
CC BY 4.0, 383 recorded actions / 346 eligible examples. Model uploads are deferred.
This is a small simulation development dataset: the original learned selector beat
random on held-out starts but both lost coverage. Reliable unfolding, material
fidelity and physical pinch grasping remain unqualified.

## Install

Use Python 3.11 or 3.12. Offline model development needs Torch, **not Genesis**:

```bash
git clone https://github.com/huw-man/cloth-fling.git
cd cloth-fling
python -m venv .venv
. .venv/bin/activate
pip install '.[train]' huggingface_hub
```

For lightweight Parquet/image readers only, use `pip install .` instead.
For NVIDIA simulation, install `pip install '.[sim]'` (Genesis 1.4.4 and Torch).
On the tested RTX 5080, use CUDA 12.8 Torch wheels:
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.
With uv, `uv sync --extra sim` selects that index automatically.

## Download dataset v1

```bash
hf download humannew/cloth-fling --type dataset --revision v1 --local-dir data/cloth-fling-v1
```

| Split | Recorded | Eligible |
|---|---:|---:|
| Train | 308 | 272 |
| Validation | 23 | 22 |
| Test | 52 | 52 |

Parquet files embed before/after RGB, executed grasp coordinates and transforms,
coverage, labels, failures and episode/reset identities. Negative gains are kept;
unsettled/ineligible examples have null supervised targets. The separate 21 fixture
bundles contain privileged simulation state and hashes, never learned inputs.
See the [dataset card](src/cloth_fling/release_assets/dataset-card.md).

## Train and evaluate offline

Every output directory must be fresh. Paths are explicit; train reads only `train`.
A short CPU install/checkpoint smoke run is:

```bash
cloth-fling policy train --dataset data/cloth-fling-v1 --output runs/cpu-model --steps 2 --device cpu
cloth-fling policy evaluate --mode offline --dataset data/cloth-fling-v1 --checkpoint runs/cpu-model/model.pt --output runs/validation
```

Omit `--steps 2` for the existing 2,000-step fit. Validation is the evaluation
default; selecting test requires `--split test`. Offline reports contain eligible
example count, MAE, Huber loss (delta .1) and prediction bias (prediction − target)
at the executed action pixel. This does not measure policy rollout success.

## Collect and evaluate in simulation

These commands require the `sim` extra, a free GPU and downloaded fixture bundles.
Collection uses only training fixtures. Evaluation defaults to validation fixtures;
`--split test` explicitly selects held-out fixtures. No resets occur inside an episode.

```bash
cloth-fling policy collect --fixtures data/cloth-fling-v1 --output runs/collection --episodes 1
cloth-fling policy train --dataset runs/collection/examples --output runs/collected-model
cloth-fling policy evaluate --mode sim --fixtures data/cloth-fling-v1 --checkpoint runs/cpu-model/model.pt --output runs/sim-validation --device cuda
```

A very small collection may contain fewer than the required 32 eligible training
examples; collect more episodes to a fresh directory when needed. Downloaded v1
already has 272 training examples. `--fixture-id` restricts simulation to one
fixture in the selected split. Defaults are at most three actions, stopping on
settled, fully contained 80% coverage, with 35 simulated / 180 wall seconds per
action and a one-hour command ceiling (`--max-wall-seconds` can reduce it).
`--max-actions 1` or `2` reduces the action cap. Placement-quality violations are
observations; physical execution, artifact, ownership and cleanup regressions stop
execution. Failed/uncertain runs retain evidence and cannot be resumed or retried
in place. Simulation reports give success within one/two/three actions, terminal
coverage, failures and actions used. GPU smoke runs validate mechanics only.

## Checkpoints and custom predictors

`model.pt` is the existing RGB U-Net 32/64/128/256 CPU state dict, loaded with
`torch.load(..., weights_only=True)`. The adjacent `model.json` records architecture,
preprocessing, frozen v4 configuration hash, weights hash, fitting parameters and
source hashes. Keep both files together; `cloth_fling.learning.load_model` verifies
them. Existing compatible pilot checkpoints use the same loader.

A custom Python `module:function` takes a NumPy **uint8 RGB tensor
`[96,64,64,3]`** and returns finite immediate signed-gain maps `[96,64,64]`.
T is rotation-major (12) then scale (8), with row/column last. No privileged inputs
are passed to this function. Environment feasibility filtering happens after prediction.

```python
# my_policy.py (install it or put its directory on PYTHONPATH)
import numpy as np

def predict(rgb):
    return np.zeros((96, 64, 64), dtype=np.float32)
```

```bash
cloth-fling policy evaluate --mode sim --fixtures data/cloth-fling-v1 --predictor my_policy:predict --output runs/custom-validation
```

The same interface works with `--mode offline --dataset data/cloth-fling-v1`.
Offline evaluation reconstructs RGB transforms from the saved RGB crop; it never
uses fixture geometry or depth. The existing learner fits immediate outcomes at
executed pixels; these whole-action examples are not ACT trajectory demonstrations.

## Development and history

```bash
pip install '.[dev,train]'
pytest -m 'not gpu'
ruff check src tests scripts
python scripts/export_dataset_v1.py --source runs/image-learning-pilot-v1 --output /tmp/cloth-fling-v1-export
```

CPU CI checks readers, policies, packaging and installation without Genesis.
The [publication record](docs/publication-v1.md) pins source/data releases and
verification. All historical commands remain available via `cloth-fling --help`.
The [previous README](docs/historical-readme.md), [runtime](docs/runtime.md),
[frozen motion](docs/frozen-laydown.md), [original learning pilot](docs/image-learning-pilot-results.md)
and other campaign documentation are retained. Their terminal runs remain closed;
local traces, videos, checkpoints and unrelated run directories are excluded from
this release. Installing this repository does not download archived run directories.
