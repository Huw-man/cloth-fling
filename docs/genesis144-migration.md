# Genesis 1.4.4 migration record

## Release identity and scope

On 2026-10-19 the official [PyPI release](https://pypi.org/project/genesis-world/1.4.4/)
and [GitHub release](https://github.com/Genesis-Embodied-AI/genesis-world/releases/tag/v1.4.4)
identified **1.4.4**, released 2026-10-11, as the latest stable Genesis release.
Upstream commit: `a649105285d30bb9cd3464e753bc5270f3f43d4b`.

The normal PyPI resolver supplies `genesis_world-1.4.4-py3-none-any.whl`.
Its SHA-256 is
`66ad14c4f2ecf8cdad8535df84ef7e1cee6aea654cbceb86288c5d1fcf451de0`,
matching both the independently verified wheel and the artifact recorded in `uv.lock`.
The lock uses a registry dependency and content hash, not a local wheel path or a
neighboring source checkout. No system/global Python installation is changed.

Project and import version advance from 0.1.0 to **0.2.0** because engine and physical
snapshot compatibility change. This is a local package boundary, without a tag or
publication. Historical Genesis 0.4.3 artifacts and the historical root `.venv` are
preserved until coordinator integration. Previous milestone acceptance does not transfer.
Cloth-motion diagnostics, primitive tuning, collection and training remain paused.

## Dependencies and interpreter

`requires-python = ">=3.11,<3.13"` remains unchanged. Genesis 1.4.4 declares Python
`>=3.10,<3.14`; its dependencies resolve across both project-supported minors, so no
Python-floor increase is required. Installation and import checks target Python 3.12.13;
a successful universal lock is not execution validation on Python 3.11.

The base extra remains NumPy, Pillow and PyArrow only. Genesis/Torch stay optional in
`sim`, and application imports remain lazy. Genesis does not declare Torch, so the
project retains an explicit Torch dependency and the explicit `pytorch-cu128` index.
CUDA 12.8 is the intended RTX 5080 software target, not evidence of target-GPU execution.

Selected locked/installed versions for Python 3.12 on Linux x86_64:

| Package | Version |
| --- | --- |
| cloth-fling | 0.2.0 |
| genesis-world | 1.4.4 |
| quadrants | 1.3.3 |
| torch | 2.11.0+cu128 |
| trimesh | 4.12.2 |
| pydantic | 2.14.1 |
| gs-madrona | 0.0.10 |
| av | 19.0.1 |
| numpy | 2.5.4 |
| pillow | 11.3.0 |
| pyarrow | 23.0.1 |
| imageio / imageio-ffmpeg | 2.38.1 / 0.6.0 |

Genesis's wheel requires Quadrants exactly 1.3.3, Pydantic >=2.13.0, Trimesh >=4.8.2,
Pillow >11.0, gs-madrona 0.0.10 on Linux x86_64, and other declared packages including
PyAV, DracoPy and xacro. The resolver now locks those declarations rather than retaining
the 0.4.3 closure. The lock branches NumPy to 2.4.6 and PyAV to 18.1.0 for Python 3.11.
The project raises its own Trimesh lower bound to 4.8.2 to match the engine requirement.
No unneeded dependency is added to the offline base.

## Shared migration environment

The coordinator supplies a shared environment under ignored `runs/genesis144/venv`,
separate from the historical root `.venv`. From the assigned dependency worktree,
with `UV_PROJECT_ENVIRONMENT` set to that absolute environment directory, installation is:

```bash
UV_CACHE_DIR=/tmp/cloth-fling-uv-cache \
  uv sync --locked --python 3.12 --extra sim --extra dev --extra video
```

Use the shared environment's `bin/python` directly for checks in another worktree, with
`PYTHONPATH="$PWD/src"` selecting that worktree's source. The editable install initially
points to the dependency worktree; do not let it silently select the wrong agent's code.
Avoid concurrent `uv sync` mutations while other agents use the environment.

Read-only sandbox imports may need writable dependency caches:

```bash
export NUMBA_CACHE_DIR=/tmp/cloth-fling-numba-cache
export MPLCONFIGDIR=/tmp/cloth-fling-matplotlib
```

These variables relocate caches; they do not change the engine, contact law or GPU policy.
The root historical environment is not upgraded by this installation.

## Contact, snapshots and rendering

The upgraded CLI default is `SceneConfig(contact_profile="native_coulomb_v2")`, supplied
by the runtime/contact integration, with the matching `configs/contact-native-v2.json`.
Explicit historical JSON is parsed with its original semantics; it is not silently
rewritten into v2. The backend must reject unsupported old physical profiles.
Library `SceneConfig()` retains historical config identity for offline artifacts.

Genesis 1.4.4's legacy PBD/rigid coupling still omits tangential cloth friction. Upstream
rigid multi-contact improvements do not supply the missing cloth-contact law. The
runtime agent owns a guarded v2 extension and fresh contact qualification. Adding a
version to an old private-API whitelist cannot establish compatibility. Preserve failed
probes, and keep all physical acceptance thresholds/materials/damping unchanged.

Physical snapshots move to schema 2 with engine/runtime identity. Physical restore must
reject schema-1 and mismatched engine/configuration/topology before physical writes.
Historical Parquet, images and safe NPZ/JSON remain readable; offline verification and
`replay --inspect` never initialize simulation. Rendered historical replay needs an
explicit recorded-geometry-only hook, exact compatible rest topology and particle order,
and separate source/runtime/rendering metadata. It must not silently restore old physics.

`runtime_provenance` records installed package/engine versions without importing Genesis
or Torch, together with code revision, dirty state and lock hash. Installed distribution
identity is distinct from backend-reported runtime identity. `config_provenance` records
the contact profile, backend identity/capabilities, independent rendering identity and
capabilities, and source snapshot schema/engine/configuration identity. Missing capability
reports remain null, and missing historical engine metadata remains unknown. No capability
or old score is promoted to new-runtime qualification. Source hashes and original RGB
remain distinct from new rendered frames.

## Validation and handoff

Dependency integration performed only resolver, installation, import and offline checks.
The Genesis import check did not call `gs.init`: `gs._initialized` remained false.
Torch reported `torch.version.cuda == "12.8"`; no CUDA availability query or scene was
used as proof. `uv lock --check` passed and `uv pip check` found all 113 installed
packages compatible. A separate base/dev environment contains neither Genesis nor Torch.
The historical root environment still reported Genesis 0.4.3 at this check.

CLI/provenance tests depend on the runtime agent's v2 contract. Release checks used
temporary copied source trees, without importing an adjacent checkout or committing
another agent's owned files:

- Release code plus the v2 contract: base/dev suite **287 passed, 2 skipped, 3 GPU
  deselected**; shared sim environment's release/offline tests **26 passed**.
- Release code plus runtime commits `ae557c8`, `b45e6fc` and rendering commit `5aac3b7`:
  composed base/dev suite **318 passed, 2 skipped, 3 GPU deselected**.
- Repository Ruff checks and `git diff --check` passed. Wheel and source distribution
  built as 0.2.0; wheel metadata verified the optional exact Genesis pin, Python range
  and matching import version.

These are CPU/source/storage checks, not initialized-runtime or GPU qualification.
The coordinator must run the integrated checks again after bringing the branches together.
Offline regression coverage includes historical schema-1 dataset reading, negative
rewards, safe state decoding and replay inspection with Genesis/Torch imports explicitly
blocked. Python 3.11 execution remains unchecked in this dependency handoff.

Only the coordinator can grant an exclusive RTX 5080 lease after dependency installation
and CPU readiness. Scene/render, native-contact, restore and behavior acceptance are
separate gates; none is inherited from the old milestones. The dependency agent runs no
GPU scenes, primitive campaigns, collection, training, hardware, cloud or publication.
Bounded future runtime checks are in [runtime.md](runtime.md); historical behavior and
failed evidence are preserved in [milestone2-results.md](milestone2-results.md).
