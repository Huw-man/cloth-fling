# Stage 1 primitives and diagnostic

Stage 1 commands two assisted attachment points through `contracts.ClothBackend`.
These modules import NumPy and contracts, with Pillow imported only when saving
PNG evidence. They never import Genesis or torch. No arms, IK, pinch grasping,
training, hardware execution, cloud work or broad campaigns are implemented.
A successful execution means the bounded lifecycle completed with finite state;
`behavior_passed` remains `null`. The CPU fake tests do not qualify Genesis,
physical contact grasping, or unfolding quality.

## Public calls

```python
from cloth_fling.selection import (
    SelectionConfig, NoValidPairError, feasible_candidates, select_action,
)
from cloth_fling.sim.primitives import PrimitiveConfig, execute_action
from cloth_fling.diagnostics import run_diagnostic

# SceneConfig is the normal selection configuration; SelectionConfig exposes search budgets.
action = select_action(
    observation, env.config, policy="uniform_valid", primitive="laydown", rng=rng,
)
outcome = execute_action(env, action, config=PrimitiveConfig())

# Diagnostic uses the current fixture. The caller may reset or restore it beforehand.
report = run_diagnostic(
    env, seed=0, output_dir=output_dir, primitive_config=PrimitiveConfig(), action=None,
)
```

Exact signatures:

- `select_action(observation: Observation, config: SceneConfig | SelectionConfig, *, policy: str = 'uniform_valid', primitive: str = 'laydown', rng: numpy.random.Generator | None = None) -> GraspAction`
- `feasible_candidates(observation: Observation, config: SceneConfig | SelectionConfig, *, primitive: str = 'laydown', max_pairs: int = 256) -> list[GraspAction]`
- `execute_action(env: ClothBackend, action: GraspAction, config: PrimitiveConfig | None = None) -> ActionOutcome`
- `run_diagnostic(env: ClothBackend, *, seed: int = 0, output_dir=None, primitive_config: PrimitiveConfig | None = None, action: GraspAction | None = None) -> dict`

`execute_action` never resets/restores. Valid actions can follow one another on the
evolving cloth. It captures pre-action RGB/depth observation and actual backend
coverage, calls `attach(action.pixels)`, and commands only attachment targets.
Each command is followed by exactly one `env.step()`, with positions and velocities
read afterward. Backend step duration must match `env.config.sim_dt`.
No command writes particle positions, velocities, damping or material parameters.
The backend must resolve each visible pixel to its surface and preserve velocity
when releasing. Distinct returned particle IDs and the attachment lifecycle are checked.

## Selection and calibration agreement

World coordinates are metres with z up. Pixels are `(u,v)`, top-left origin, u right,
v down. `SceneConfig.grasp_workspace_m` defines a square centered at world x=y=0;
separation uses 3D Euclidean distance. Distinct pixels must be at least two pixels
apart. Depth/calibration are privileged execution feasibility inputs. Only
`Observation.policy_rgb()` is a learned-policy input.

The primary schema agreed with Stage 0 is a perspective overhead camera:

```python
observation.calibration = {
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "camera_to_world": [[...], [...], [...], [0, 0, 0, 1]],
    "image_size": [width, height],
    "pixel_center_offset": 0.5,
}
```

Depth is positive optical-z metres. With `o=pixel_center_offset` (default 0.5),
`q = depth * inv(K) @ [u+o, v+o, 1]` in the OpenCV camera frame; the homogeneous
camera-to-world transform maps q into world coordinates. `pixels_to_world`
implements this unprojection and rejects missing, invalid or unsupported calibration.
An explicit orthographic CPU-fixture schema is also supported:
`world_bounds_m=[xmin,xmax,ymin,ymax]`, `camera_height_m=H`, with
`x=xmin+u/(W-1)*(xmax-xmin)`, `y=ymax-v/(Himg-1)*(ymax-ymin)`, `z=H-depth`.
The Genesis path uses the perspective schema.

The foreground mask is RGB-derived: uint8-scale maximum channel >45 **and**
chroma (`maximum-minimum`) >20. It supports uint8 RGB and float RGB in [0,1].
This rejects the known gray table `(0.25,0.28,0.30)` as well as black background.
The scene must use a visibly saturated cloth; gray cloth or a colored/textured
background requires a new foreground method. Depth never supplies the foreground
mask. Before either policy samples or ranks pairs, `conservative_foreground_mask`
erodes the RGB mask by one full 3x3 neighborhood, including diagonals; image borders
count as background. Thus each chosen pixel has eight foreground neighbors. One-
and two-pixel-wide fragments are rejected. Geometry-guided ranking uses the boundary
of this eroded mask. This reduces silhouette/ray disagreement but cannot correct
stale or mismatched rendered cloth geometry. Stage 0/coordinator should compare
the rendered mask with ray-hit cloth pixels during GPU qualification; the CPU
gray-table fixture is not that check.

`uniform_valid` samples distinct foreground pixel pairs uniformly by rejection
with at most 4096 attempts, then checks depth, separation and workspace.
Both policies reserve an explicit surface-to-vertex snap uncertainty of
`r = 1.5 * config.mesh_spacing_m` for each endpoint. Selected surface points must
satisfy `min_separation + 2*r <= distance <= max_separation - 2*r`, and each world
x/y coordinate must satisfy `abs(coordinate) <= grasp_workspace_m/2 - r`.
The pair margin accounts for both endpoints snapping in opposite directions;
the workspace margin applies to each endpoint individually. The default spacing
is 0.01 m (r=0.015 m); `SceneConfig.mesh_spacing_m` is propagated unchanged.
These margins can leave no feasible pair, which is reported explicitly. Given an
explicit RNG, sampling is reproducible; the default RNG seed is 0. Exhausting the
budget raises `NoValidPairError`, even if an unsearched feasible pair might exist.
`geometry_guided` chooses the widest feasible RGB boundary pair with deterministic
pixel lexicographic tie-break. It uniformly subsamples image-order boundary pixels
to at most 512 before pair enumeration. `feasible_candidates` returns the widest
feasible pairs from that bounded set; an empty result means no pair there.
The returned geometry baseline policy label is
`geometry_guided_privileged_depth`. No hidden particle state is used for selection.

## Motion and bounds

The controlled baseline phases are:
`attach -> lift -> tauten -> diagonal_laydown -> drag -> release -> settle`.
The candidate phases are:
`attach -> lift -> tauten -> swing -> pullback -> lower -> place -> release -> settle`.

Lift is one continuous trajectory. Tautening reads the measured grasp geometry
and shortest-rest-path sag every step. It uses a smooth span-rate envelope, gated
by measured remaining span and sag, with bounded rate and acceleration. If available,
rest positions and mesh edges/faces supply a deterministic Dijkstra path and
rest-edge extension measurements. Span is capped by both scene max separation and
`rest_geodesic_distance * (1 + max_strain)`. Target span uses `taut_rest_ratio`.
The graph distance depends on mesh triangulation and is an upper bound on a smooth
surface geodesic; it is not a tearing model. Every step additionally checks the
maximum tensile edge extension against `max_strain`. Optional metadata can be
`env.rest_positions`, `env.edges`/`env.faces` or the equivalent keys in `snapshot()`.
Missing metadata uses bounded endpoint spreading and explicitly reports
`strain_bound_available=false`; this fallback has no material strain guarantee.
Malformed metadata or disconnected grasp vertices fail explicitly.

Laydown moves diagonally down/forward and continues into a short near-table drag.
The fling candidate retains attachment throughout one continuous swing, pullback,
lowering and placement trajectory. Release checks actual grasp z <= table+0.02 m.
Both use C2 piecewise quintic Bezier interpolation with shared tangents through
internal waypoints. Derivative control hulls analytically bound speed/acceleration;
time is lengthened as needed. Dense physics samples introduce no stops. The entire
lift, tauten and final trajectory have intended stopping boundaries; swing/pullback/
lower/place and laydown/drag internal boundaries pass through without stopping.
Actual command finite differences are checked too. Workspace violations fail
explicitly; trajectories are never silently clipped into a different motion.

PrimitiveConfig defaults (SI):

| Field(s) | Default |
| --- | --- |
| `table_height_m`, `lift_height_m`, `release_height_m` | 0, 0.30, 0.008 |
| `lift_seconds`, `tauten_seconds` | 1.8, 2.0 |
| `laydown_seconds`, `drag_seconds`, `fling_segment_seconds` | 2.0, 0.8, 0.45 |
| `laydown_distance_m`, `drag_distance_m` | 0.12, 0.06 |
| `swing_distance_m`, `pullback_distance_m`, `place_distance_m` | 0.16, 0.10, 0.04 |
| `max_speed_m_s`, `max_acceleration_m_s2` | 1.4, 5.0 |
| `tauten_speed_m_s`, `fallback_spread_m` | 0.08, 0.08 |
| `max_strain`, `taut_rest_ratio`, `taut_sag_tolerance_m` | 0.10, 0.98, 0.008 |
| `settle_rms_m_s`, `settle_hold_seconds`, `settle_timeout_seconds` | 0.01, 0.5, 10.0 |
| `max_simulated_seconds`, `max_wall_seconds` | 30.0, 120.0 |
| `trace_interval_seconds` | 0.05 |

Settling requires `sqrt(mean(sum(v_xyz**2))) < 0.01 m/s` continuously for 0.5 seconds;
any threshold crossing restarts the quiet window. It is capped at 10 seconds.
Config may shorten durations, distances, settling windows and deadlines for smoke
tests; the report records the exact config. Hard action caps are 30 simulated and
120 wall seconds, settling timeout <=10 s, threshold <=0.01 m/s, release height
<=0.02 m. Deadlines are checked around bounded single-step backend calls. Python
cannot interrupt an indefinitely blocked backend call; host supervision is needed
for backend hangs. Mandatory release and terminal observation capture still run
after an error/deadline, so their wall overhead may extend past the deadline.

## Outcomes and evidence

`ActionOutcome` uses unchanged shared contracts. Signed coverage gain is
`coverage_after - coverage_before`, also in `diagnostics['coverage_gain']`;
negative gains are retained. `coverage()` supplies actual backend projected cloth
area and flat reference area. A changing flat reference is invalid.

Invalid actions/errors return `execution_valid=false`, an explicit `reason`,
`terminated=true`, and detailed typed errors. Deadlines set `truncated=true`.
Release is attempted even after partial attachment or unexpected exceptions.
A release that changes any cloth velocity is invalid. Finite positions/velocities
are checked throughout. Settling timeout is invalid. The terminal observation is
captured after cleanup; if capture fails, the outcome retains a marked fallback
observation, unavailable coverage as NaN, and
`diagnostics['terminal_observation_valid']=false`. Consumers must check that flag.
A failed initial observation/coverage raises `PrimitiveFailure` after cleanup,
since no honest contract-compatible before observation can be supplied.

`evidence['phases']` records phase ordering. `evidence['phase_traces']` samples the
full measured cloth positions/velocities, commanded and measured grasp positions,
active attachment IDs, actual coverage, max edge strain, RMS/max speed and table
proximity. Phase boundaries force additional samples. Grasp positions continue
being measured after release. Terminal state arrays are recorded separately.
Full coverage is measured at trace cadence and action endpoints; finite state,
edge strain, speed, attachment state and deadlines are checked every physics step.
Diagnostics record command speed/acceleration maxima, release velocity delta,
release coverage, post-release recoil, peak sampled settle-coverage recoil,
settling and metadata availability. Recoil is a signed coverage drop from release
or a nonnegative drop from the sampled settle peak; it is not a force measurement.
Edge extension, rest-path sag and z-near-table fraction carry explicit proxy labels.
Table proximity is not native contact and attachment is not a pinch-grasp proof.
All geometry/state/depth evidence is privileged and excluded from learned RGB input.

## Bounded comparison and artifacts

`run_diagnostic` selects a favorable visible geometry-guided pair before motion,
or takes a manually reviewed `action` pair. It holds those exact pixels fixed for
one laydown baseline and one fling candidate on independent restores of the same
current fixture. The caller owns initial reset/fixture choice. The seed is recorded;
it does not silently replace the current fixture. The diagnostic never calls reset.
It verifies exact positions, velocities, attachment IDs, simulation clock and
coverage before each action and after restoring the original fixture on exit.
A mismatch stops the comparison and invalidates restoration. Physical state is
verified directly; stochastic renderer noise is not included in the state hash.
It compares final cloth coverage, signed gain, strain, settling and recoil.
Hand paths do not establish behavior quality. `quality_qualified=false` and
`behavior_passed=null` remain explicit even when both executions complete.
The framework is bounded to two action attempts with a cooperative total wall
budget of twice the per-action budget, plus initial/final capture/restore overhead.
The 600-run favorable-grasp campaign is deferred.

With `output_dir`, relative artifact references in the report point to:

- `diagnostic.json`: JSON-safe report, traces and outcome measurements (NaN becomes null).
- `fixture.json` and `fixture.npz`: recursive snapshot metadata with `npz_key`
  references to NumPy arrays; no pickle/object arrays.
- `{laydown,fling}_{before,after}.png`: actual RGB observations for each primitive.
- `{laydown,fling}_depth.npz` and `{laydown,fling}_camera.json`: privileged optical
  depth and calibration support for both observations.

The JSON report is atomically replaced. Artifact write failures are explicit.
The two-run comparison is not transactional dataset collection; use the Stage 2
tools for repeated-action ancestry and artifact validation.

## Research provenance and verification

[FlingBot paper, section 3.2 and real-world stretch description](https://proceedings.mlr.press/v164/ha22a/ha22a.pdf)
describes lift/stretch, forward motion, pullback, placement and release, and uses
visible cloth geometry as a stretch proxy. The inspected source revision is
`c5f35c785c6bc5d49282fb10117aa66ca04bf144`:
[`environment/simEnv.py`](https://github.com/real-stanford/flingbot/blob/c5f35c785c6bc5d49282fb10117aa66ca04bf144/environment/simEnv.py)
(`stretch_cloth`, `fling_primitive`, `pick_and_fling_primitive`) retains grasp
through its lowering/place commands;
[`real_world/fling.py`](https://github.com/real-stanford/flingbot/blob/c5f35c785c6bc5d49282fb10117aa66ca04bf144/real_world/fling.py)
uses blended swing/touchdown moves before opening the grippers. This package is
a smaller robotless attachment-assisted adaptation with new bounded interpolation
and measurement, not a reproduction of the learned policy or reported results.

CPU tests use a 25-particle fake with gravity, distance constraints, table collision,
friction, surface RGB/depth rasterization and two attachment endpoints. They verify
lifecycle, continuous commands and bounds, negative gain, closed-loop spreading,
rest-strain failure, release momentum, failed/partial attachment, nonfinite and
unexpected errors, deadlines, continuous settling, no reset across valid actions,
fixed-pair restore validity and image/state artifacts. Selection tests cover the
primary perspective schema, gray-table masking, bounded search, deterministic
geometry ties and distinct feasible pixels without backend access.

```sh
UV_CACHE_DIR=/tmp/cloth-fling-uv-cache uv run --extra dev pytest tests/test_primitives.py tests/test_selection.py
```

Stage 0 holds the GPU qualification slot. This Stage 1 work runs no GPU jobs.
The coordinator must validate the integrated Genesis primitives on the RTX 5080,
including rendered mask/ray-hit agreement, release momentum, timing, actual
strain/coverage/recoil, restoration, and short repeated-action collection.
