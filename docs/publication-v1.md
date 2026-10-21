# Dataset and policy workflow v1

The public workflow adds `policy collect`, `policy train`, and `policy evaluate`
(`offline`/`sim`) around the existing immediate RGB value-map learner and frozen
`reference_post_grasp_laydown_v4`. Historical commands and the previous README
remain in the source history and `docs/historical-readme.md`.

Dataset: https://huggingface.co/datasets/humannew/cloth-fling (CC BY 4.0), tagged
`v1`. Code: https://github.com/huw-man/cloth-fling (MIT). Model weights are not
released. The dataset's `release-manifest.json` pins the exporter code revision,
original executed revision, source artifact hashes, frozen v4 hash and all export
file hashes. The export includes only the completed original image-learning pilot:
308/23/52 recorded train/validation/test actions, 272/22/52 eligible labels, 21
separate hashed fixture bundles. All negative outcomes and 37 null labels remain.
No single-reset or family-pilot data is included.

Verification uses the full CPU suite, additional RGB/action/label and split/hash
checks, a clean wheel install with Torch and no Genesis, CPU fitting/checkpoint
reload/offline evaluation, a full reachable-history scan and an allowlist-based
release scan. The two bounded native smoke actions verify collection/evaluation
mechanics, not policy qualification. Final measured receipts and pinned release
revisions are recorded after execution.

The source implementation lives on the isolated `publish/workflow` branch. It
does not change or restart the active family pilot; its run roots, source identity,
GPU lease and historical terminal evidence remain separately owned. Native smoke
execution acquires the same exclusive file lock only after it becomes free.
