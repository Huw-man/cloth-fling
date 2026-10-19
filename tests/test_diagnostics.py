"""CPU restored-fixture and privileged review checks, no Genesis execution."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import test_primitives as lifecycle
from test_primitives_fake import FakeClothBackend

from cloth_fling.diagnostics import reviewed_pair_candidates, run_diagnostic
from cloth_fling.selection import conservative_foreground_mask, feasible_candidates
from cloth_fling.sim.mesh import finite_radius_fold, resolve_grasps, square_mesh


@pytest.fixture
def short_config():
    return lifecycle.short_config.__wrapped__()


@pytest.mark.parametrize("record_video", [False, True])
@pytest.mark.parametrize("primitives", [("laydown",), ("fling",), ("fling", "laydown")])
def test_requested_primitives_restore_exact_fixture(
    short_config, tmp_path, primitives, record_video
):
    recorders = []

    class NumericRecorder:
        def __init__(self, output_dir, **kwargs):
            self.calls = self.closes = 0
            recorders.append(self)

        def start_action(self, index):
            self.index = index

        def __call__(self, backend, trace):
            assert "cloth_centroid_m" in trace and "attachment_error_max_m" in trace
            self.calls += 1

        def close(self, *, complete):
            self.closes += 1
            return {"complete": complete}

    env = FakeClothBackend()
    fixture = copy.deepcopy(env.snapshot())
    # An unrequested primitive's video directory must not block a single run.
    if len(primitives) == 1:
        other = ({"laydown", "fling"} - set(primitives)).pop()
        (tmp_path / "video" / f"diagnostic-{other}").mkdir(parents=True)
    report = run_diagnostic(
        env,
        primitives=primitives,
        primitive_config=short_config,
        output_dir=tmp_path,
        record_video=record_video,
        recorder_factory=NumericRecorder,
    )
    assert report["execution_valid"], report
    assert [r["primitive"] for r in report["runs"]] == list(primitives)
    assert env.attach_calls == len(primitives) and env.reset_calls == 0
    assert len(report["recordings"]) == (len(primitives) if record_video else 0)
    assert all(r.calls > 0 and r.closes == 1 for r in recorders)
    assert report["final_restore_valid"]
    np.testing.assert_array_equal(env.positions, fixture["positions"])
    np.testing.assert_array_equal(env.velocities, fixture["velocities"])
    assert all(
        r["grasp_particle_ids"] == report["runs"][0]["grasp_particle_ids"] for r in report["runs"]
    )
    assert all(r["pixels"] == report["fixed_pixels"] for r in report["runs"])
    if len(primitives) == 1:
        assert "candidate_minus_baseline_final_coverage" not in report
        assert "no paired comparison" in report["comparison"]
    else:
        fling, laydown = report["runs"]
        assert report["candidate_minus_baseline_final_coverage"] == pytest.approx(
            fling["coverage_after"] - laydown["coverage_after"]
        )
    assert json.loads((tmp_path / "diagnostic.json").read_text()) == report


@pytest.mark.parametrize(
    "primitives", [(), [], "laydown", ("laydown", "laydown"), ("oops",), None, (["laydown"],)]
)
def test_primitive_selection_rejected_before_backend_access(primitives):
    with pytest.raises(ValueError, match="primitives"):
        run_diagnostic(object(), primitives=primitives)


def test_reviewed_pairs_are_visible_resolver_feasible_and_material_ranked():
    env = FakeClothBackend()
    env.rest_positions, env.faces = square_mesh(size=0.2, spacing=0.01)
    env.rest_positions[:, 2] = 0.004
    env.positions = env.rest_positions.copy()
    env.velocities = np.zeros_like(env.positions)
    env.positions = finite_radius_fold(env.positions, radius=0.02, base_height=0.004)
    before = env.positions.copy(), env.velocities.copy(), env.simulated_time
    observation = env.observe()
    candidates = feasible_candidates(observation, env.config, max_pairs=256)
    rows = reviewed_pair_candidates(env, max_pairs=8)
    assert len(rows) >= 2
    assert rows == reviewed_pair_candidates(env, max_pairs=8)
    assert [r["rest_flat_chord_m"] for r in rows] == sorted(
        [r["rest_flat_chord_m"] for r in rows],
        reverse=True,
    )
    assert len({tuple(sorted(r["predicted_grasp_particle_ids"])) for r in rows}) == len(rows)
    mask = conservative_foreground_mask(observation)
    for row in rows:
        assert any(np.array_equal(row["pixels"], a.pixels) for a in candidates)
        pixels = np.array(row["pixels"], dtype=int)
        assert mask[pixels[:, 1], pixels[:, 0]].all()
        ids = resolve_grasps(
            row["pixels"],
            env.positions,
            env.faces,
            observation.calibration,
            radius=env.config.mesh_spacing_m * 1.5,
            workspace=env.config.grasp_workspace_m,
            min_separation=env.config.min_grasp_separation_m,
            max_separation=env.config.max_grasp_separation_m,
        )
        assert row["predicted_grasp_particle_ids"] == list(ids)
        assert max(row["snap_distances_m"]) <= env.config.mesh_spacing_m * 1.5
        assert row["rest_flat_chord_m"] == pytest.approx(
            np.linalg.norm(env.rest_positions[ids[1]] - env.rest_positions[ids[0]])
        )
        assert row["requires_visual_review"]
    # Ranking reads only: no attachments, commands, clock or state changes.
    assert env.attach_calls == env.step_calls == env.release_calls == 0
    np.testing.assert_array_equal(env.positions, before[0])
    np.testing.assert_array_equal(env.velocities, before[1])
    assert env.simulated_time == before[2]
    json.dumps(rows, allow_nan=False)


def test_reviewed_pairs_do_not_fall_back_to_hidden_nearest_vertices():
    env = FakeClothBackend()
    # No snap lies within the backend's unchanged 1.5 cm radius: fail closed.
    env.config = replace(env.config, mesh_spacing_m=0.00001)
    assert reviewed_pair_candidates(env) == []
    assert env.attach_calls == env.step_calls == 0


@pytest.mark.parametrize("budgets", [(0, 256), (8, 0), (True, 256), (8, 1.2)])
def test_reviewed_pairs_reject_invalid_budgets(budgets):
    with pytest.raises(ValueError):
        reviewed_pair_candidates(object(), max_pairs=budgets[0], candidate_budget=budgets[1])
