---
license: cc-by-4.0
language:
- en
task_categories:
- robotics
size_categories:
- n<1K
pretty_name: Cloth-fling v1
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---

# Cloth-fling v1

383 simulated grasp/action outcomes from the completed original image-learning
pilot, using a synthetic 0.60 × 0.40 m towel, overhead RGB and two assisted native
vertex pins. Each selected grasp executes the frozen v4 lift/tauten/hold/laydown,
centered 12 cm drag, release and settling continuation. This is immediate-value
RGB grasp learning, not a continuous-control or physical pinch-grasp dataset.

| Split | Recorded actions | Eligible supervised examples |
|---|---:|---:|
| Train | 308 | 272 |
| Validation | 23 | 22 |
| Test | 52 | 52 |
| Total | 383 | 346 |

## Download and use

```bash
hf download humannew/cloth-fling --type dataset --revision v1 --local-dir data/cloth-fling-v1
cloth-fling policy train --dataset data/cloth-fling-v1 --output runs/my-model --steps 10
cloth-fling policy evaluate --mode offline --dataset data/cloth-fling-v1 --checkpoint runs/my-model/model.pt --output runs/my-validation
```

See [code and installation](https://github.com/huw-man/cloth-fling) for the `train`
and `sim` dependency extras, custom predictors and simulation examples. No model
weights are included. The `v1` tag pins this release; `release-manifest.json` gives
exact source artifact, code, frozen configuration and exported file hashes.

## Fields and supervision

`before_rgb` and `after_rgb` embed original PNG bytes in `{bytes, path}` structs.
Images are uint8 RGB at 256 × 256. `action_u0/v0/u1/v1` are the two original-image
grasp endpoints, with top-left origin, u right and v down. `map_index_tvu` stores
transform, row, column in a 96 × 64 × 64 action map. `to_original` and
`action_encoding_json` retain the exact executed transform, angle/scale and RGB
crop. Replaying this transform gives a 64 × 64 RGB model input; transform index is
12 rotations × 8 scales. The target is immediate signed coverage gain at the
executed map pixel. Other pixels are unlabelled.

`coverage_before/after/gain`, `reward` and measured areas retain original outcomes,
including negative gains. `supervised_target` is null for all 37 ineligible rows;
never replace it with zero. `eligible` requires valid execution, settled outcome,
a recorded encoding and executed feasibility. `settled`, `execution_valid`,
`failed`, `reason`, `terminated`, `truncated` and `end_reason` preserve eligibility
and failure context. `failed` marks invalid or unsettled actions, not every action
with a negative gain.

`episode_key`, `run_id`, `episode_id`, `step_id`, `transition_id`, reset IDs/hashes,
`reset_group_id`, `seed` and observation IDs preserve grouping and exact chains.
Whole fold families remain in one split; validation/test each include the saved
paired random and learned evaluations. Do not fit on validation/test, split rows
randomly or treat repeated episodes as independent reset families.

The 21 frozen schema-2 fixture JSON/NPZ bundles live separately in `fixtures/`,
with `fixtures-manifest.json`. They contain privileged state for simulation reset,
feasibility and evaluation only. The predictor receives RGB, never state, depth,
coverage, feasibility masks or material geometry. No full diagnostic traces,
videos, single-reset overfit data or near-half-fold pilot data are included.

## Evidence and limitations

The original held-out learner reached mean terminal coverage .476394 versus
.351837 for random selection, but both reduced mean initial coverage .622646.
Reliable unfolding remains unmet. There are 37 settlement failures and a peak
local edge ratio of 2.970057. Material fidelity and timestep-dependent contact
behavior remain unqualified. This dataset supports studying this small simulated
value-learning loop, not claims of generalization or hardware performance.
The original collection used its historical .95 coverage stopping rule; new
`policy` commands default to settled, fully contained .80 stopping within three
actions. No resetting occurs between actions of an episode.

## Reproduction and attribution

From the source checkout, with the original local archived run available:

```bash
python scripts/export_dataset_v1.py --source runs/image-learning-pilot-v1 --output /tmp/cloth-fling-v1-export
```

The exporter refuses other/incomplete campaigns, verifies source data, counts,
labels, splits and fixtures, and writes fresh outputs. Source traces remain local;
source manifest hashes permit identifying the archived provenance. Export Parquet
byte identity depends on the recorded PyArrow version; row identities and RGB,
transform and label parity are verified independently.

Dataset copyright 2026 Huw Newman. Licensed **CC BY 4.0**; see `LICENSE`.
Attribute “Huw Newman, Cloth-fling dataset v1” and link to
https://huggingface.co/datasets/humannew/cloth-fling, cite the revision used, and
indicate changes. The code is separately MIT licensed at the GitHub repository.
