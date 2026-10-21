# Dataset and policy workflow v1

[Code](https://github.com/huw-man/cloth-fling) is MIT licensed.
[Dataset v1](https://huggingface.co/datasets/humannew/cloth-fling/tree/6ea839399e250e1f3010b457ba8e376fcd1e0a96)
is CC BY 4.0, with the `v1` tag pinned to
`6ea839399e250e1f3010b457ba8e376fcd1e0a96`. Model uploads are deferred.

The workflow adds `policy collect`, `policy train`, and `policy evaluate`
(`offline`/`sim`) around the existing immediate RGB value-map learner and frozen
`reference_post_grasp_laydown_v4`. The previous README is retained in
`docs/historical-readme.md`; historical commands and terminal campaigns remain intact.
The source changes were made in the isolated `publish/workflow` worktree. The
family pilot's source, run roots and execution were not modified or interrupted.
Its coordinator's subsequent closing report is included in the retained history.

## Pinned provenance

- Exporter/package revision: `833cf5e1b47e27b1657f604fe77afed86a4d1633`.
- Original pilot executed revision: `347c6403918a2e5fc19bd8adbb56a98ef44b9e88`.
- Frozen v4 specification SHA-256:
  `6a5caa7c01afa0735882b575c197c513556f7e8903fd23e8fe7c8d6334278339`.
- Published release manifest SHA-256:
  `f6eff550029a2f7d5f86116a90d47fc0603a0fccace3fd20d3b3634336f5eb33`.

The dataset's `release-manifest.json` pins every exported file and the source
artifact manifests. Only the completed original image-learning pilot was exported:
308/23/52 recorded train/validation/test actions, 272/22/52 eligible labels, 144
episode groups and 21 separate schema-2 fixture bundles. All 221 eligible negative
gains remain signed; the 37 ineligible outcomes retain measurements and null
supervised targets. No single-reset, family-pilot or publication-smoke actions
are included. Full traces, videos, checkpoints and unrelated run directories stay local.
The 48 payload files total 10,176,476 bytes; the small release manifest is additional.

## Verification

- **610 CPU tests passed**, eight native tests deselected; Ruff and diff checks passed.
- A second export reproduced all 48 payload files byte for byte. The only manifest
  change identified the later committed exporter revision.
- All 383 rows independently matched source RGB bytes/hashes, grasp coordinates,
  saved transforms/indices, signed labels, eligibility and episode/reset identities.
  Split isolation and all fixture JSON/NPZ/state hashes passed.
- Base-wheel readers and the frozen profile loaded with neither Torch nor Genesis
  installed. A separate clean train-extra installation used Torch 2.14.1+cpu without
  Genesis, fitted two steps on 272 eligible examples, reloaded its checkpoint and
  evaluated 22 validation examples. Both checkpoint and custom `module:function`
  prediction paths produced finite MAE, Huber and bias reports. These are install
  checks, not a new model-quality result; the temporary checkpoint is not released.
- Anonymous `hf download ... --revision v1` succeeded with token variables removed
  and an isolated empty HF home. The downloaded manifest, all file hashes, split
  counts and 21 fixture bundles passed offline verification without Genesis.
- All local branches were scanned for credential patterns and unintended artifacts.
  No findings were present. `scripts/audit_publication.py` reproduces the history
  scan and release allowlist check. Historical development branches are retained.

GitHub CPU CI covers Python 3.11/3.12, base/train wheel installation and bundled
profile loading. Final anonymous code-install/quickstart and hosted CI results are
recorded with the `workflow-v1` GitHub release.
Frozen schedule tests compare recomputed polynomial coefficients with 1e-12
absolute tolerance across NumPy/BLAS builds. Structure, timing, configuration
bytes and controller hashes retain exact checks; no runtime or physics was changed.

## Two native smoke actions

After the pilot released the exclusive lease, exactly two actions ran under the
same lock in **82.4596 seconds total**, below the shared ten-minute ceiling. There
were no retries. Both used the identical released fixture bytes and frozen v4
physics/motion; both completed, settled, released pins with velocity preserved,
and independently certified cleanup. The lease was released after each command.

| Command | Split / fixture | Actions | Terminal coverage | Failures |
|---|---|---:|---:|---:|
| `policy collect` | train / `fold-50-+00` | 1 | .302233 | 0 |
| `policy evaluate --mode sim` | validation / `fold-30-+00` | 1 | .992067 | 0 |

Evaluation used the existing original-pilot initial checkpoint through the normal
loader. Native source revision was `2c52e2c`; subsequent changes count unfinalized
failed attempts in reports and improve packaging/CI. No additional native actions
were run. Local peak edge ratios were 1.648078 and 1.243959. These smokes verify
mechanics and reports on selected fixtures, not policy, contact or material
qualification. The original pilot's reliable-unfolding goal remains unmet.

Local evidence is preserved under
`.worktrees/publish-workflow/runs/publication-smoke-v1/` relative to the project
root. Replays/viewer panes were not opened. No model release, cloud training,
hardware execution or broader collection was performed.
