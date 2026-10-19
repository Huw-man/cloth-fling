"""Opt-in, bounded host qualification; never run a GPU scene implicitly."""

import json
import os
import time

import numpy as np
import pytest

from cloth_fling.contracts import SceneConfig
from cloth_fling.sim.backend import GenesisBackend, InvalidResetError, _state_hash
from cloth_fling.sim.coverage import CoverageGrid
from cloth_fling.sim.mesh import pixel_ray, ray_surface_hit

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("CLOTH_FLING_RUN_GPU") != "1",
                                 reason="set CLOTH_FLING_RUN_GPU=1 for exclusive host GPU run")]


def pixels_for(points, calibration):
    pose = np.linalg.inv(np.asarray(calibration["camera_to_world"]))
    camera = np.c_[points, np.ones(len(points))] @ pose.T
    uvz = camera[:, :3] @ np.asarray(calibration["intrinsics"]).T
    return tuple(map(tuple, uvz[:, :2] / uvz[:, 2:] - 0.5))


def foreground(rgb):
    return (rgb.max(axis=2) > 45) & (np.ptp(rgb.astype(np.int16), axis=2) > 20)


def erode_one_pixel(mask):
    padded = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    return np.logical_and.reduce([padded[y:y + h, x:x + w]
                                  for y in range(3) for x in range(3)])


def nonflat_render_metrics(backend, observation):
    """Independent projected triangle union plus ray depths on interior samples."""
    positions = backend.get_positions()
    calibration = observation.calibration
    inverse_pose = np.linalg.inv(np.asarray(calibration["camera_to_world"]))
    camera = np.c_[positions, np.ones(len(positions))] @ inverse_pose.T
    assert np.all(camera[:, 2] > 0)
    uvz = camera[:, :3] @ np.asarray(calibration["intrinsics"]).T
    width, height = calibration["image_size"]
    assert width == height
    projected = np.c_[uvz[:, :2] / uvz[:, 2:] - width / 2, np.zeros(len(positions))]
    # Cell centres are exactly u+.5,v+.5; this oracle uses PHYSICS triangles,
    # not the Genesis visual mesh, rendered depth, or RGB segmentation.
    oracle = CoverageGrid(width, 1).mask(projected, backend.faces)
    predicted = foreground(observation.rgb)
    interior_prediction = erode_one_pixel(predicted)
    interior_oracle = erode_one_pixel(oracle)
    precision = float(np.mean(oracle[interior_prediction]))
    recall = float(np.mean(predicted[interior_oracle]))
    raw_iou = float(np.sum(predicted & oracle) / np.sum(predicted | oracle))
    assert precision > .995 and recall > .995
    assert raw_iou > .94
    candidates = np.argwhere(interior_prediction & interior_oracle)
    errors = []
    for v, u in candidates[np.linspace(0, len(candidates) - 1, 128, dtype=int)]:
        origin, direction = pixel_ray((u, v), calibration)
        hit, _ = ray_surface_hit(origin, direction, positions, backend.faces)
        expected_depth = (inverse_pose @ [*hit, 1])[2]
        errors.append(abs(float(observation.depth[v, u]) - expected_depth))
    quantiles = np.percentile(errors, [50, 95, 100])
    # Genesis skins the 1 cm visual mesh from neighboring remeshed particles;
    # allow half a mesh spacing, not centimetres of stale planar depth.
    assert quantiles[-1] < backend.config.mesh_spacing_m / 2
    return {"eroded_precision": precision, "eroded_recall": recall, "raw_iou": raw_iou,
            "predicted_pixels": int(predicted.sum()), "oracle_pixels": int(oracle.sum()),
            "depth_error_m_median_p95_max": quantiles.tolist()}


def test_nonflat_first_frame_and_immediate_restore(tmp_path):
    """Fold must be the FIRST rendered state, to catch unallocated GL buffers."""
    report = {}
    stamp = time.monotonic()
    with GenesisBackend(SceneConfig()) as backend:
        original_render = backend.camera.render
        render_calls = []

        def immutable_render(**options):
            state = backend.snapshot()
            step = backend.scene._t
            result = original_render(**options)
            assert backend.scene._t == step
            assert backend.snapshot()["state_hash"] == state["state_hash"]
            render_calls.append(options)
            return result

        backend.camera.render = immutable_render
        folded = backend.reset(11, kind="fold")
        fold_state = backend.snapshot()
        assert len(render_calls) == 2  # Warm buffers, then upload actual folded vertices.
        report["fold_first_frame"] = nonflat_render_metrics(backend, folded)
        # Direct observation/restore cannot hide a refresh behind physics steps.
        step = backend.scene._t
        original_step = backend.scene.step

        def forbidden_step(*args, **kwargs):
            pytest.fail("observation/restore must not advance physics")

        backend.scene.step = forbidden_step
        # A shifted, raised NONPLANAR fixture at the same step exercises explicit
        # setters through restore, with substantial silhouette and depth changes.
        shifted = {key: value.copy() if isinstance(value, np.ndarray) else value
                   for key, value in fold_state.items()}
        shifted["positions"] += np.array([-.12, -.08, .12], dtype=np.float32)
        shifted["state_hash"] = _state_hash(shifted)
        raised = backend.restore(shifted)
        report["same_step_raised_restore"] = nonflat_render_metrics(backend, raised)
        assert backend.scene._t == step
        assert backend.snapshot()["state_hash"] == shifted["state_hash"]
        assert np.mean(foreground(raised.rgb) != foreground(folded.rgb)) > .02
        restored = backend.restore(fold_state)
        np.testing.assert_array_equal(restored.rgb, folded.rgb)
        np.testing.assert_array_equal(restored.depth, folded.depth)
        assert restored.observation_id == folded.observation_id
        assert backend.snapshot()["state_hash"] == fold_state["state_hash"]
        assert backend.scene._t == step
        backend.scene.step = original_step
        flat = backend.reset(7, kind="flat")
        assert np.mean(foreground(flat.rgb) != foreground(folded.rgb)) > .05
        # Restore folded state after a flat render, at an unchanged scene step.
        backend.scene.step = forbidden_step
        step = backend.scene._t
        from_flat = backend.restore(fold_state)
        report["fold_restore_after_flat"] = nonflat_render_metrics(backend, from_flat)
        assert backend.scene._t == step
        assert backend.snapshot()["state_hash"] == fold_state["state_hash"]
        np.testing.assert_array_equal(from_flat.rgb, folded.rgb)
        np.testing.assert_array_equal(from_flat.depth, folded.depth)
        assert from_flat.observation_id == folded.observation_id
        assert backend.attached_ids == () and backend._free().all()
        np.savez_compressed(tmp_path / "nonflat-render-evidence.npz", rgb=folded.rgb,
                            depth=folded.depth, **fold_state)
        report["wall_seconds"] = time.monotonic() - stamp
        print("STAGE0_NONFLAT_RENDER_REPORT=" + json.dumps(report, sort_keys=True))


def test_host_pipeline(tmp_path):
    start = time.monotonic()
    report = {}
    with GenesisBackend(SceneConfig()) as backend:
        report["build_seconds"] = time.monotonic() - start
        report["particles"] = len(backend.rest_positions)
        report["faces"] = len(backend.faces)
        before = backend.reset(7, kind="flat")
        flat_state = backend.snapshot()
        report["flat_reset"] = backend.reset_metadata.copy()
        assert before.rgb.dtype == np.uint8 and before.rgb.std() > 10
        assert before.depth.dtype == np.float32 and np.isfinite(before.depth).all()
        assert backend.observe().observation_id == before.observation_id
        assert backend.coverage() == pytest.approx((1, .25, .25), abs=.002)
        chroma = np.ptp(before.rgb.astype(np.int16), axis=2)
        cloth_pixels = before.depth < 1.797
        table_pixels = (before.depth > 1.799) & (before.depth < 1.801)
        report["rgb_segmentation"] = {
            "cloth_chroma_min_median_max": np.percentile(
                chroma[cloth_pixels], [0, 50, 100]).tolist(),
            "table_chroma_min_median_max": np.percentile(
                chroma[table_pixels], [0, 50, 100]).tolist(),
            "table_rgb_max": int(before.rgb[table_pixels].max()),
            "chroma_gt30_precision": float(np.mean(cloth_pixels[chroma > 30])),
            "chroma_gt30_recall": float(np.mean(chroma[cloth_pixels] > 30)),
        }
        assert report["rgb_segmentation"]["chroma_gt30_precision"] > .99
        assert report["rgb_segmentation"]["chroma_gt30_recall"] > .99
        foreground = (before.rgb.max(axis=2) > 45) & (chroma > 20)
        ray_matches = []
        flat_positions = backend.get_positions()
        for expected in (True, False):
            locations = np.argwhere(foreground == expected)
            for v, u in locations[np.linspace(0, len(locations) - 1, 128, dtype=int)]:
                origin, direction = pixel_ray((u, v), before.calibration)
                try:
                    ray_surface_hit(origin, direction, flat_positions, backend.faces)
                    hit = True
                except ValueError:
                    hit = False
                ray_matches.append(hit == expected)
        report["rgb_segmentation"]["ray_agreement_256_balanced_samples"] = float(
            np.mean(ray_matches))
        assert np.mean(ray_matches) > .99
        json.dumps(before.calibration, allow_nan=False)
        p = backend.get_positions()
        # Two separated interior material points, projected using actual calibration.
        requested = np.array([[-.15, 0, 0], [.15, 0, 0]])
        candidates = [int(np.argmin(np.linalg.norm(p - point, axis=1))) for point in requested]
        pixels = pixels_for(p[candidates], before.calibration)
        ids = backend.attach(pixels)
        report["attached_ids"] = ids
        assert ids == tuple(candidates)
        assert (~backend._free()).sum() == 2
        # Instrument public setters: only the two attached IDs may be commanded.
        calls = []
        original_velocity = backend.cloth.set_particles_vel
        original_position = backend.cloth.set_particles_pos

        def checked_velocity(values, particles_idx_local=None, **kwargs):
            calls.append(tuple(particles_idx_local) if particles_idx_local is not None else None)
            return original_velocity(values, particles_idx_local, **kwargs)

        def no_position_writes(*args, **kwargs):
            pytest.fail("motion attempted a particle position overwrite")

        backend.cloth.set_particles_vel = checked_velocity
        backend.cloth.set_particles_pos = no_position_writes
        initial = backend.get_positions()[list(ids)]
        endpoint_errors = []
        for i in range(1, 31):
            target = initial + [0.0001 * i, 0, 0.001 * i]
            backend.set_attachment_targets(target)
            backend.step()
            endpoint_errors.append(float(np.max(np.abs(
                backend.get_positions()[list(ids)] - target))))
        assert calls and set(calls) == {ids}
        assert max(endpoint_errors) < 2e-5
        report["max_endpoint_error_m"] = max(endpoint_errors)
        velocity = backend.get_velocities()[list(ids)]
        np.testing.assert_allclose(velocity, np.tile([.05, 0, .5], (2, 1)), atol=2e-4)
        attached_snapshot = backend.snapshot()
        with pytest.raises(ValueError, match="attached"):
            backend.restore(attached_snapshot)
        backend.release()
        report["release_velocity_m_s"] = backend.get_velocities()[list(ids)].tolist()
        np.testing.assert_array_equal(backend.get_velocities()[list(ids)], velocity)
        assert backend.attached_ids == () and backend._free().all()
        backend.cloth.set_particles_vel = original_velocity
        backend.cloth.set_particles_pos = original_position
        backend.step()
        assert np.isfinite(backend.get_positions()).all()
        assert np.linalg.norm(backend.get_velocities()[list(ids)]) > .1
        state = backend.snapshot()
        observation = backend.observe()
        path = tmp_path / "snapshot.npz"
        np.savez_compressed(path, **state)
        backend.step(5)
        with np.load(path, allow_pickle=False) as loaded:
            restored = backend.restore(dict(loaded))
        np.testing.assert_array_equal(backend.get_positions(), state["positions"])
        np.testing.assert_array_equal(backend.get_velocities(), state["velocities"])
        assert backend.attached_ids == () and backend._free().all()
        assert backend.simulated_time == state["simulated_time"]
        assert restored.observation_id == observation.observation_id
        # Seeded inputs repeat; GPU constraint reductions need not be bitwise deterministic.
        repeat = backend.reset(7, kind="flat")
        delta = float(np.max(np.abs(backend.get_positions() - flat_state["positions"])))
        report["repeated_flat_max_position_delta_m"] = delta
        report["repeated_flat_bitwise_observation_identity"] = (
            repeat.observation_id == before.observation_id)
        assert delta < 1e-5
        for kind in ("drop", "fold"):
            stamp = time.monotonic()
            try:
                backend.reset(11, kind=kind)
            except InvalidResetError:
                # A failed preparation must be represented honestly. This gate
                # qualifies failure reporting, not a random drop distribution.
                if kind != "drop":
                    raise
                backend.observe()  # Invalid terminal state remains observable.
            report[kind] = {"wall_seconds": time.monotonic() - stamp,
                            "coverage": backend.coverage(), **backend.reset_metadata}
            assert backend.attached_ids == () and backend._free().all()
            assert np.isfinite(backend.get_positions()).all()
            if backend.reset_metadata["status"] == "valid":
                assert backend.reset_metadata["stable_seconds"] >= .5
                assert backend.reset_metadata["final_rms_speed_m_s"] < .01
            else:
                assert backend.reset_metadata["status"] == "invalid"
                assert backend.reset_metadata["reason"]
        # Bunch preparations are evidence for these seeds only, not a reset-bank distribution.
        report["total_wall_seconds"] = time.monotonic() - start
        print("STAGE0_GPU_REPORT=" + json.dumps(report, sort_keys=True))
