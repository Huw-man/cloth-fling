# Visual capture and saved-trajectory replay

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
build. Merely setting Genesis surface options is insufficient in Genesis 0.4.3:
its default SimpleMaterial conversion can lose the requested PBR values.

## Backend integration (contact/coordinator owned)

No physical contract or snapshot schema change is needed. Keep the constructor
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

Keep appearance metadata separate from the physical snapshot. Old snapshots keep
their original physical hash/schema. Canonical physical config hashing is sorted
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
unrelated debug objects remain. A `try/finally` restores camera transform and its
redundant native pose fields. This path never calls `observe`, changes the cached
policy RGB/depth, steps physics, or requests debug depth. Debug objects are removed
before subsequent clean policy rendering, preventing marker pixels/shadows/depth
from entering the learner view. Backend native camera recording must be disabled
for this scoped capture (use the recorder below). Native GPU isolation still needs
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

Metadata includes each video's hash, dimensions, nominal fps, frame count,
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
`--appearance legacy` (default) preserves old table pixels; a JSON path also works.
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
physics snapshot attachment flags are cleared for restoration. Replay replaces
`env.step/reset` and `scene.step/reset` with guards that raise. It restores saved
mesh states and renders, with no reset generator or physical re-execution.

Choose a fresh output directory **outside** the source dataset/diagnostic directory.
Outputs include the videos above, original before/after observation PNG copies,
and `replay-provenance.json` with source manifest/diagnostic hashes, fixture hashes,
invalid outcomes, truncation status, original observation references and per-video
hashes. Reconstructed outputs explicitly say `reconstructed_replay`,
`original_video_recorded=false`, `physics_reexecuted=false`, and
`captured_during_execution=false`. `--max-frames` bounds rendering and labels
truncation. These are newly rendered saved trajectories, never original execution
video. Old physical snapshot/config compatibility remains the backend owner's
responsibility; unknown physical fields are not silently discarded.

## Validation scope

CPU tests cover deterministic table texture and actual material readback validation,
marker measurement/release behavior, exception cleanup, unchanged policy caches,
camera/state preservation, writer lifecycle, both legacy formats, no-step guards,
source immutability and provenance. GPU visual review requires the coordinator's
exclusive lease after the contact gate; CPU tests do not establish native renderer
or RTX 5080 qualification. Review the actual integrated output before accepting
appearance and rendering isolation.
