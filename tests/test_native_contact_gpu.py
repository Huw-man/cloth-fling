"""Opt-in bounded tests of the actual native Quadrants contact function."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_fling.contracts import SceneConfig
from cloth_fling.sim.mesh import square_mesh, write_obj
from cloth_fling.sim.native_contact import ENGINE_VERSION, install_native_contact

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("CLOTH_FLING_RUN_GPU") != "1", reason="exclusive host GPU lease required"
    ),
]


@contextmanager
def native_contact_scene(config, directory):
    """Direct 1.4.4 physics coupon; independent of the separate rendering port."""
    import genesis as gs

    assert gs.__version__ == ENGINE_VERSION
    if not gs._initialized:
        gs.init(backend=gs.cuda, precision="32", seed=0, logging_level="warning")
    assert gs.backend == gs.cuda and gs.device.type == "cuda"
    rest, faces = square_mesh(config.cloth_size_m, config.mesh_spacing_m)
    path = directory / "coupon.obj"
    write_obj(path, rest, faces)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.sim_dt, substeps=4),
        rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
        coupler_options=gs.options.LegacyCouplerOptions(),
        pbd_options=gs.options.PBDOptions(
            particle_size=config.mesh_spacing_m,
            max_stretch_solver_iterations=config.max_stretch_solver_iterations,
        ),
        show_viewer=False,
    )
    try:
        table = scene.add_entity(
            gs.morphs.Box(size=(1.2, 1.2, 0.04), pos=(0, 0, -0.02), fixed=True)
        )
        scene.add_entity(
            gs.morphs.Mesh(file=str(path), pos=(0, 0, config.mesh_spacing_m / 2 + 0.001)),
            material=gs.materials.PBD.Cloth(
                rho=config.areal_density_kg_m2,
                stretch_compliance=config.stretch_compliance,
                bending_compliance=config.bending_compliance,
                static_friction=config.static_friction,
                kinetic_friction=config.kinetic_friction,
                air_resistance=config.air_resistance,
            ),
        )
        coupler = install_native_contact(scene, table, config)
        scene.build()
        yield SimpleNamespace(
            scene=scene,
            particle_diameter=config.mesh_spacing_m,
            substeps=4,
            take_contact_accounting=coupler.take_contact_accounting,
        )
    finally:
        scene.destroy()


def test_native_impulse_microtests(tmp_path):
    import quadrants as qd
    from genesis.engine.couplers.legacy_coupler import LegacyCoupler
    from genesis.utils import array_class

    # Exercise native SDF/normal response and subclass friction with identical input.
    @qd.kernel
    def probe(
        c: qd.template(),
        inputs: qd.types.ndarray(),
        out: qd.types.ndarray(),
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_info: array_class.RigidInfo,
        collider_static_config: qd.template(),
    ):
        for k in range(inputs.shape[0]):
            p = qd.Vector([0.0, 0.0, inputs[k, 0]])
            v = qd.Vector([inputs[k, 1], inputs[k, 2], inputs[k, 3]])
            c.pbd_solver.particles_reordered[k, 0].ipos = p - v * c.pbd_solver._substep_dt
            c.pbd_solver.particles_reordered[k, 0].free = inputs[k, 4] > 0
            normal_old = qd.Vector([0.0, 0.0, 1.0])
            p0, v0, n0 = LegacyCoupler._func_pbd_collide_with_rigid_geom(
                c,
                k,
                p,
                v,
                1.0,
                normal_old,
                0,
                0,
                geoms_state,
                geoms_info,
                links_state,
                sdf_info,
                rigid_info,
                collider_static_config,
            )
            p1, v1, n1 = c._func_pbd_collide_with_rigid_geom(
                k,
                p,
                v,
                1.0,
                normal_old,
                0,
                0,
                geoms_state,
                geoms_info,
                links_state,
                sdf_info,
                rigid_info,
                collider_static_config,
            )
            for axis in qd.static(range(3)):
                out[k, 0, axis] = p0[axis]
                out[k, 1, axis] = v0[axis]
                out[k, 2, axis] = p1[axis]
                out[k, 3, axis] = v1[axis]
                out[k, 4, axis] = n0[axis]
                out[k, 5, axis] = n1[axis]

    cfg = SceneConfig(
        contact_profile="native_coulomb_v2", cloth_size_m=0.1, mesh_spacing_m=0.02, image_size=32
    )
    with native_contact_scene(cfg, tmp_path) as env:
        radius, dt = env.particle_diameter / 2, cfg.sim_dt / env.substeps
        inputs = np.array(
            [
                [radius - dt, 1.0, 0, -1.0, 1],  # impact: dv_n=1, dv_t=.3
                [radius - dt, 0.1, 0, -1.0, 1],  # static stop
                [radius - 9.81 * dt * dt, 0.5, 0, -9.81 * dt, 1],  # resting load
                [radius + 0.1, 0.5, 0.2, -1.0, 1],  # airborne, no impulse
                [radius - dt, 0.5, 0.2, -1.0, 0],  # attached, no friction
                [radius - dt, -1.0, 0, -1.0, 1],  # reversed impact
            ],
            dtype=np.float32,
        )
        out = np.empty((len(inputs), 6, 3), dtype=np.float32)
        env.take_contact_accounting(reset=True)
        env.scene.sim.coupler.clear_contact_trace()
        env.scene.sim.coupler._begin_contact_substep()
        rigid = env.scene.rigid_solver
        probe(
            env.scene.sim.coupler,
            inputs,
            out,
            rigid.dyn_state.geoms,
            rigid.dyn_info.geoms,
            rigid.dyn_state.links,
            rigid.collider._sdf._sdf_info,
            rigid.rigid_info,
            rigid.collider.collider_config,
        )
        np.testing.assert_array_equal(out[:, 0], out[:, 2])
        np.testing.assert_allclose(out[:, 1, 2], out[:, 3, 2], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(out[:, 4], out[:, 5])
        np.testing.assert_allclose(out[:2, 3, 0], [0.7, 0], atol=2e-5)
        np.testing.assert_allclose(
            out[2, 1, 0] - out[2, 3, 0], 0.3 * 9.81 * dt, rtol=0.003, atol=1e-6
        )
        np.testing.assert_allclose(out[3:5, 1], out[3:5, 3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(out[5, 3, 0], -out[0, 3, 0], atol=2e-5)
        assert np.all(np.sum(out[:, 3] ** 2, axis=1) <= np.sum(out[:, 1] ** 2, axis=1) + 1e-7)
        accounting = env.scene.sim.coupler.read_contact_trace()
        totals = dict(zip(accounting["columns"], accounting["values"].sum(axis=0), strict=True))
        assert totals["attached_top_contacts"] == 1
        assert totals["noncontact_calls"] == 1
        assert totals["free_friction_events"] == 4
        assert totals["attached_friction_events"] == 0
        assert totals["noncontact_friction_events"] == 0
        assert totals["friction_work_j"] < 0
        assert totals["positive_friction_work_j"] == 0
        np.savez_compressed(
            tmp_path / "native-microtests.npz",
            inputs=inputs,
            outputs=out,
            native_contact_values=accounting["values"],
            native_contact_columns=np.array(accounting["columns"]),
        )
        first = env.take_contact_accounting()
        assert first["substeps"] == 1
        assert first["totals"]["attached_top_contacts"] == 1
        env.scene.sim.coupler.stop_contact_trace()
        env.scene.sim.coupler._begin_contact_substep()
        again = np.empty_like(out)
        probe(
            env.scene.sim.coupler,
            inputs,
            again,
            rigid.dyn_state.geoms,
            rigid.dyn_info.geoms,
            rigid.dyn_state.links,
            rigid.collider._sdf._sdf_info,
            rigid.rigid_info,
            rigid.collider.collider_config,
        )
        np.testing.assert_array_equal(again, out)  # Logging cannot affect physics.
        second = env.take_contact_accounting()
        assert second["substeps"] == 2
        for column in first["totals"]:
            assert second["totals"][column] == pytest.approx(2 * first["totals"][column])
        assert env.scene.sim.coupler.read_contact_trace()["substeps"] == 1
        env.scene.reset()
        assert env.take_contact_accounting() == second
        assert env.take_contact_accounting(reset=True) == second
        cleared = env.take_contact_accounting()
        assert cleared["epoch"] == second["epoch"] + 1
        assert cleared["substeps"] == 0 and not any(cleared["totals"].values())
        # Installation affects this scene only, never the upstream class.
        assert type(env.scene.sim.coupler).__base__ is LegacyCoupler
        assert LegacyCoupler._func_pbd_collide_with_rigid_geom is not (
            type(env.scene.sim.coupler)._func_pbd_collide_with_rigid_geom
        )
