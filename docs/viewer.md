# Visual capture and saved-trajectory replay

## Genesis 1.4.4 migration status

Rendering targets the stable 1.4.4 wheel (2026-10-11, upstream `a649105`),
not the old project `.venv`. Use the coordinator-prepared shared latest-runtime
Python with `PYTHONPATH=$PWD/src`; do not import the inspection wheel as a runtime.
CPU checks are separate from native GPU qualification. No 0.4.3 visual/contact or
behavior qualification carries forward. Cloth-motion campaigns remain paused.
GPU scenes require the coordinator's exclusive lease.

### Native-source audit and retained adapters

The inspected release sources are [PBD entities](https://github.com/Genesis-Embodied-AI/genesis-world/blob/a649105/genesis/engine/entities/pbd_entity.py),
[PBD render fields](https://github.com/Genesis-Embodied-AI/genesis-world/blob/a649105/genesis/engine/solvers/pbd_solver.py),
[rasterizer context](https://github.com/Genesis-Embodied-AI/genesis-world/blob/a649105/genesis/vis/rasterizer_context.py),
[depth shader dispatch](https://github.com/Genesis-Embodied-AI/genesis-world/blob/a649105/genesis/ext/pyrender/renderer.py),
[material conversion](https://github.com/Genesis-Embodied-AI/genesis-world/blob/a649105/genesis/utils/mesh.py),
and [camera](https://github.com/Genesis-Embodied-AI/genesis-world/blob/a649105/genesis/vis/camera.py).

- Native PBD still copies/remeshes the input visual mesh for physics and computes
  visual vertices by weighted particle supports. Native rendering is therefore
  not guaranteed to draw exact physical triangles through folds. Retain a detached
  physics-topology visual copy and identity supports. The 1.4.4 adapter checks
  entity/offset/batch layout, copy independence, exact vertex/face order, field
  shapes, device readback and unchanged physical state. No whitelist permits the
  old 0.4.3 adapter to run on a new version.
- Native depth-only dispatch still forces backface culling and selects
  `mesh_depth.vert` without a two-sided geometry shader. Retain the camera-local
  segmentation-program substitution for double-sided depth. It keeps the native
  pixel-centre transform and depth readback; no depth rescaling is introduced.
  Inspect the installed shader-dispatch source before patching and reject an
  unreviewed dispatch. Folded-ray correctness still requires a leased GPU check.
- Native non-emissive image material conversion still uses `SimpleMaterial`
  without the requested PBR factors. Retain the table-visual-only PBR/UV adapter
  and inspect actual native roughness .8, metallic 0, texture and UVs after build.
  Rigid visual geometry now resolves through its descriptor; its public `vmesh`
  still exposes the exact trimesh passed into native rasterization.
- Native 1.4.4 raster cameras skip markers unless `camera.debug` is true. Scope
  `_debug=True` to synchronous RGB-only debug capture and restore it on success
  and failure, including batched pose tensors. Reject raytracer/batch-renderer
  cameras. Marker primitives remain native debug objects, excluded from shadow
  passes. Never request marker depth or modify policy caches.

No adapter was removed based on a version assumption. The existing initial draw
is retained as buffer-upload warmup pending native qualification; there is no
same-step cache patch here, only native `force_render=True`. Public 1.4.4 setters
and native rendering refresh are used without a physics step.

`PhysicsMeshRendering.identity` and `.capabilities` are fresh JSON-safe
dictionaries, forwarded by the runtime as `env.rendering_identity` and
`env.rendering_capabilities`. Identity includes the actual engine version,
adapter revision, exact-triangle topology, support/depth mechanisms and native
renderer class names read from installed camera targets. Unknown class names
and uninstalled mechanisms remain `None`. Installation flags/counts are updated
only after layout/state/device readback checks succeed.

These dictionaries describe configuration separately from qualification.
`qualification_status="unqualified"`, all `*_qualified=false` flags and
`qualification_inherited=false` remain explicit; successful installation never
promotes native GPU, RGB/depth rays, same-step restore, folded backfaces, debug
isolation or matte appearance to qualified. Those require separately recorded
leased checks. Provenance and new video engine metadata include both dictionaries;
absent backend metadata stays `None`.

These tools produce privileged debug/evaluation artifacts. Policy input remains the
original RGB observation. Debug markers, geometry, depth, overlays and clearance
must never enter learned RGB input. Appearance applies to the **table only**;
cloth surface, collision material, friction, particles and trajectories are unchanged.

`AppearanceConfig()` selects neutral gray matte laminate: RGB (.55,.55,.55),
roughness .8, metallic 0, with subtle seeded achromatic texture. Load the same
profile with `AppearanceConfig.from_json("configs/appearance-matte.json")`.
`appearance=None` preserves the milestone1 dark table; `AppearanceConfig.legacy()`
selects it explicitly. Inputs are validated, including finite unit-range channels,
bounded grain amplitude and integer texture dimensions/seed. The noise generator
is local, so appearance cannot perturb the physics random generator.

The native table raster material is PBR, with a real color texture and UVs. The
adapter verifies the actual native material roughness, metallic and texture after
build. Merely setting Genesis surface options is insufficient in Genesis 1.4.4:
its default SimpleMaterial conversion can lose the requested PBR values.

## Backend integration (contact/coordinator owned)

Physical snapshots now use runtime-owned schema 2 with engine identity; 0.4.3
schema 1 cannot be physically restored. Keep the constructor
keyword `appearance: AppearanceConfig | None = None`. Preserve cloth construction.

```python
from cloth_fling.sim.appearance import (
    make_surface, install_visual_texture, install_native_material,
)

self.appearance = appearance
self.table = self.scene.add_entity(
    gs.morphs.Box(size=(config.table_size_m, config.table_size_m, .04),
                  pos=(0, 0, -.02), fixed=True),
    surface=make_surface(gs, appearance),
)
# After entity construction, BEFORE build. Only table.vgeoms[].vmesh visual changes.
if appearance is not None:
    install_visual_texture(self.scene, self.table, gs, appearance)
self.scene.build()
if appearance is not None:
    install_native_material(self.scene, self.table, gs, appearance)
```

Keep appearance metadata separate from the physical snapshot. Archived snapshots keep
their original physical hash/schema and are imported only as recorded geometry. Canonical physical config hashing is sorted
JSON with separators `(',', ':')`, excluding appearance. Video metadata records
that independent hash and the selected appearance for each output video. A visual
appearance override during replay changes re-rendered pixels, not source physics.

## Native debug capture

```python
from cloth_fling.rendering import render_debug_views
views = render_debug_views(env, phase="pre_attach", selected_targets_m=targets_xyz)
```

`views` contains overhead and oblique native RGB arrays. The overhead view uses
the backend's original camera; oblique uses a fixed z-up pose. Two red/blue native
debug spheres have **10 mm radius**. Before attachment they indicate selected XYZ
surface targets; while attached they use **measured cloth vertices** indexed by
`env.attached_ids`, never commanded hand positions. Once attachments are empty in
release/settle/terminal phases, no spheres are drawn. Historical trace grasp fields
do not keep markers visible. Native markers are debug objects, never entities.

Run capture synchronously on the scene owner's thread, outside `observe()`.
Every created marker has scoped cleanup, including partial creation/render failures;
unrelated debug objects remain. A `try/finally` restores camera transform, debug flag and its
redundant native batched pose fields, even when rendering fails. This path never calls `observe`, changes the cached
policy RGB/depth, steps physics, or requests debug depth. Debug objects are removed
before subsequent clean policy rendering, preventing marker pixels/shadows/depth
from entering the learner view. Backend native camera recording must be disabled
for this scoped capture (use the recorder below). If `GS_DISABLE_OFFSCREEN_MARKERS`
is present (even set to `0`), or Genesis already loaded with its disable flag set,
marker capture fails explicitly before drawing. Start a fresh process without that
variable; the tool never silently claims disabled markers were visible.
Native GPU isolation still needs
a bounded qualification run on the integrated backend.

## Recording during execution

The primitive owner adds optional `execute_action(..., trace_callback=None)`.
Invoke `trace_callback(env, trace)` synchronously for **sampled traces only**, after
the trace is assembled. Include a `pre_attach` trace with
`selected_grasp_positions_m` after selecting targets and before attaching. The
callback must see current `env.attached_ids` and current measured vertices. Force
an immediate post-release trace with empty attachment IDs. Callback failures must
be explicit execution failures; never swallow them.

```python
from cloth_fling.rendering import VideoRecorder

with VideoRecorder("runs/review/native", fps=20, appearance=env.appearance,
                   particle_radius_m=env.particle_diameter / 2) as recorder:
    recorder.start_action(1)
    outcome = execute_action(env, action, trace_callback=recorder)
    evidence_video_reference = recorder.metadata()  # hashes finalize after close
# Use recorder.metadata() here for finalized hashes / complete flag.
```

Create or reuse one recorder across consecutive valid actions; call
`start_action(step_id)` for each new action without resetting the simulation.
An optional `selected_targets_m` argument to `start_action` supplies pre-attachment
markers when the caller already has the selected surface targets. Live marker
positions always come from the current backend attachments.

Install optional dependencies through the coordinator-owned `video` extra:
`imageio>=2.36,<3` and `imageio-ffmpeg>=0.6,<1`. Importing modules or using offline
inspection does not import either package, Genesis or torch. MP4 encoding loads
imageio/FFmpeg only on the first appended frame. `FrameWriter` validates fixed,
even RGB frame dimensions, copies input buffers, and closes on exceptions.

Outputs are `overhead.mp4`, `oblique.mp4`, `combined.mp4`, `preview.png`, and
`video-metadata.json`. Individual panels are 384 x 384; combined is 768 x 384.
Overlays show capture type, action, phase, coverage and minimum cloth clearance:
`min(vertex_world_z) - table_height - particle_radius`. This is a geometric
clearance proxy, not a native-contact claim. Negative values and coverage losses
are preserved. Render failures propagate and mark recorder metadata incomplete.

Metadata includes source and renderer engine identities, runtime generation,
`runtime_qualification_inherited=false`, each video's hash, dimensions, nominal fps, frame count,
`native_render_capture` type, physical config hash and appearance. Actual trace
timestamps are recorded: this sampled callback cannot recover states between
traces, and phase boundary samples can change nominal playback timing. For smoother
capture the trace interval must resolve the requested fps. No quality/behavior
qualification follows from the presence of a video.

Store these outputs outside the dataset manifest initially; evidence may reference
this separate metadata and the hashed MP4 files. Do not retrospectively add debug
frames as policy RGB or modify a committed dataset manifest.

## Replay CLI

```bash
python -m cloth_fling.replay runs/dataset --inspect
python -m cloth_fling.replay runs/dataset --episode-id 0 \
  --output runs/review/replay --fps 20 --appearance matte --max-frames 2000
python -m cloth_fling.replay runs/diagnostic/diagnostic.json \
  --output runs/review/diagnostic-replay --fps 20
```

The module CLI returns 0 on success and 1 for source/rendering errors; argument
errors return argparse's code 2. It prints metadata and never opens a browser.
`--inspect` validates either format on CPU without constructing a simulator.
`--appearance legacy` (default) selects the old dark table appearance;
new-runtime pixels are newly rendered and need not equal old-runtime pixels; a JSON path also works.
The installed top-level CLI may delegate to `cloth_fling.replay.main(argv)`;
`render_replay(...)` returns the same metadata for Python callers.

Dataset input uses committed Parquet rows, evidence supports (including NPZ-backed
arrays), and saved snapshots. Rows are ordered by episode then step, retaining
invalid attempts and their reasons. Diagnostic input accepts the original
`cloth_fling.diagnostic.v1` JSON plus `fixture.json`/`fixture.npz` references, loaded
without pickle and with traversal checks. Empty trajectories render an explicitly
labeled static terminal state; they never invent action dynamics. Missing initial
samples, malformed positions, invalid attachment IDs, reversed/nonfinite times,
and topology mismatch fail clearly. Source outputs remain byte-for-byte untouched.

Replay uses the **last recorded mesh at or before each requested frame time**,
with no interpolation. Duplicate phase timestamps select the last recorded trace.
The final sample is included. Saved attachment IDs only control debug visibility;
the saved free/velocity fields are not restored as physics. The public
`cloth_fling.replay.render_recorded_state(env, source_state, trace)` wrapper uses
runtime-owned `env.restore_recorded_geometry(source_state)` to validate the
untouched source hash, exact faces/edges/rest vertices and mesh identity, then
latch that backend into render-only mode. Subsequent frames use the native public
`cloth.set_particles_pos` setter and forced native rendering. The wrapper checks
exact position readback; there is no interpolation, remapping or physical snapshot
schema conversion. Missing faces/rest vertices or different triangulation/order
fail with a specific error before import. A different new remesh requires another
explicit raw-mesh renderer; none is silently substituted here.

The built-in backend factory constructs a 1.4.4 template with
`native_coulomb_v2`, retaining the recorded geometry/material/config values.
This is an explicit render-template choice; its contact law is never executed.
Injected test/custom backend factories own their template configuration. Metadata
records untouched `source_configs` separately from `render_template_configs`
(and the existing `physics_configs` compatibility field). The source engine comes
from `runtime_identity_json` in schema 2; schema 1 is labelled Genesis 0.4.3.
The renderer engine comes from the loaded backend, never the source snapshot.
Replay guards `env.step/reset/restore` and `scene.step/reset`; the runtime latch
also rejects physical reuse. The constructor and geometry hook must never step
or generate a reset. Physical schema-2 restore remains independent and rejects
archival schema 1.

Choose a fresh output directory **outside** the source dataset/diagnostic directory.
Outputs include the videos above, original before/after observation PNG copies,
and `replay-provenance.json` with source manifest/diagnostic hashes, fixture hashes,
invalid outcomes, truncation status, original observation references and per-video
hashes. Reconstructed outputs explicitly say `reconstructed_replay`,
`original_video_recorded=false`, `physics_reexecuted=false`, and
`captured_during_execution=false`. `--max-frames` bounds rendering and labels
truncation. These are newly rendered saved trajectories, never original execution
video. Archival physical state compatibility is never claimed; unknown physical fields
are not silently discarded or imported as new-runtime physical state.

## Validation scope

CPU tests cover deterministic table texture and actual material readback validation,
marker measurement/release behavior, exception cleanup, unchanged policy caches,
camera/state preservation, writer lifecycle, both legacy formats, no-step guards,
source immutability and provenance. GPU visual review requires the coordinator's
exclusive lease after the contact gate; CPU tests do not establish native renderer
or RTX 5080 qualification. Review the actual integrated output before accepting
appearance and rendering isolation.

### Historical Genesis 0.4.3 qualification (milestone 2 only)

The following results are archival. They do not qualify Genesis 1.4.4. Existing
video artifacts and provenance remain unchanged by the migration.

On the integrated native-contact backend, the exclusive RTX 5080 check passed
with Genesis 0.4.3 on `cuda:0`. It constructed a flat static scene and **never
stepped physics or ran a primitive**. Native PBR readback verified roughness .8,
metallic 0, texture presence and UVs. Both overhead and oblique frames contained
red and blue marker pixels for selected targets and attached measured vertices;
both colors disappeared after release. Snapshot hashes, camera transform and
policy-cache identity were unchanged by every capture. A clean render after all
debug passes matched the original policy RGB and depth bitwise, and release kept
velocity exact.

The review montage is `runs/visual-native-v1/native-review.png`; individual frames,
`qualification.json`, `native-smoke.py`, logs and the three-frame native static
MP4s are beside it. A phase named `lift` in this static isolation fixture means
attached-marker display only; no cloth lift or motion was executed. Actual
marker counts are 1–4 pixels per color at the declared 256 px native resolution
(the exported video panels are resized to 384 px).

Separate serial GPU checks restored and re-rendered four frames of the original
milestone1 dataset with legacy appearance and three frames of the original
diagnostic JSON/NPZ with matte appearance. Both completed with physics stepping
guarded, explicit reconstructed-replay provenance and intentional truncation.
The original dataset manifest remained unchanged. Their outputs live in
`runs/visual-native-v1/old-dataset-legacy-replay/` and
`runs/visual-native-v1/old-diagnostic-matte-replay/`. These bounded checks establish
native render/restore compatibility for those fixtures, not primitive behavior or
full-trajectory replay qualification. The visual lease was released after the
host compute-process list became empty. User visual acceptance remains a review
of the saved native frames.
