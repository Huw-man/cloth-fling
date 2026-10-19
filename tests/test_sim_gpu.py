"""Opt-in, bounded host qualification; never run a GPU scene implicitly."""

import json
import os
import time

import numpy as np
import pytest

from cloth_fling.contracts import SceneConfig
from cloth_fling.sim.backend import GenesisBackend, InvalidResetError
from cloth_fling.sim.mesh import pixel_ray, ray_surface_hit

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("CLOTH_FLING_RUN_GPU") != "1",
                                 reason="set CLOTH_FLING_RUN_GPU=1 for exclusive host GPU run")]


def pixels_for(points, calibration):
    pose = np.linalg.inv(np.asarray(calibration["camera_to_world"]))
    camera = np.c_[points, np.ones(len(points))] @ pose.T
    uvz = camera[:, :3] @ np.asarray(calibration["intrinsics"]).T
    return tuple(map(tuple, uvz[:, :2] / uvz[:, 2:] - 0.5))


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
