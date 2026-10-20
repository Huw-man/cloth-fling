"""Scene-local position-level table Coulomb contact; Genesis imports stay lazy."""

import inspect

import numpy as np

from cloth_fling.sim.native_contact import validate_native_source

PROFILE = "reference_position_coulomb_v1"
COLUMNS = (
    "normal_correction_m",
    "tangent_correction_m",
    "normal_impulse_kg_m_s",
    "tangent_impulse_kg_m_s",
    "friction_work_j",
    "free_contacts",
    "static_events",
    "sliding_events",
    "attached_friction_events",
    "positive_friction_work_j",
)


def position_coulomb(initial, predicted, native, *, free, dt, static=0.45, kinetic=0.45):
    """Flat-top analytical reference. Normal projection is supplied by the native solver."""
    initial, predicted, native = (np.asarray(p, dtype=float) for p in (initial, predicted, native))
    free = np.asarray(free, dtype=bool)
    if (
        initial.ndim != 2
        or initial.shape[1] != 3
        or predicted.shape != initial.shape
        or native.shape != initial.shape
        or free.shape != (len(initial),)
        or not all(np.isfinite(p).all() for p in (initial, predicted, native))
        or not np.isfinite([dt, static, kinetic]).all()
        or dt <= 0
        or not 0 <= kinetic <= static
    ):
        raise ValueError("invalid position-contact inputs")
    normal = np.maximum(native[:, 2] - predicted[:, 2], 0)
    tangent = native[:, :2] - initial[:, :2]
    length = np.linalg.norm(tangent, axis=1)
    eligible = free & (normal > 0)
    stick = eligible & (length <= static * normal)
    amount = np.where(eligible, np.minimum(length, kinetic * normal), 0)
    amount[stick] = length[stick]
    correction = np.zeros_like(native)
    correction[:, :2] = (
        -tangent * np.divide(amount, length, out=np.zeros_like(length), where=length > 0)[:, None]
    )
    final = native + correction
    return final, (final - initial) / dt, correction


def install_position_contact(scene, table, config):
    import genesis as gs
    import quadrants as qd
    from genesis.engine.couplers.legacy_coupler import LegacyCoupler
    from genesis.utils import array_class

    validate_native_source(gs.__version__, inspect.getsource(LegacyCoupler))
    if (
        config.contact_profile != PROFILE
        or scene.is_built
        or type(scene.sim.coupler) is not LegacyCoupler
        or len(scene.rigid_solver.entities) != 1
        or scene.rigid_solver.entities[0] is not table
        or not table.morph.fixed
        or len(table.geoms) != 1
    ):
        raise RuntimeError(
            "position contact requires an unbuilt single fixed-table reference scene"
        )
    mu_s, mu_k = (
        np.sqrt(
            np.asarray([config.static_friction, config.kinetic_friction])
            * config.table_coefficients
        )
        if config.table_friction
        else (0.0, 0.0)
    )

    @qd.data_oriented
    class PositionTableCoupler(LegacyCoupler):
        def __init__(self, simulator, options):
            super().__init__(simulator, options)
            self.mu_static, self.mu_kinetic = float(mu_s), float(mu_k)

        def build(self):
            super().build()
            self.contact_totals = qd.field(
                qd.f64, shape=(self.pbd_solver._n_particles, self.pbd_solver._B, len(COLUMNS))
            )
            # Indexed by original particle ID, despite native spatial reordering.
            self.interval = qd.field(qd.f64, shape=(self.pbd_solver._n_particles, 9))
            self.steps = qd.field(qd.i64, shape=())
            self.original_id = qd.field(
                gs.qd_int, shape=(self.pbd_solver._n_particles, self.pbd_solver._B)
            )
            self.contact_totals.fill(0)
            self.interval.fill(0)
            self.steps.fill(0)

        @qd.kernel
        def _begin(self):
            self.steps[None] += 1
            for original, batch in qd.ndrange(self.pbd_solver._n_particles, self.pbd_solver._B):
                self.original_id[
                    self.pbd_solver.particles_ng[original, batch].reordered_idx, batch
                ] = original

        def couple(self, f):
            self._begin()
            super().couple(f)

        def take_contact_accounting(self, *, reset=False):
            a = self.contact_totals.to_numpy().sum(axis=(0, 1))
            if not np.isfinite(a).all():
                raise RuntimeError("nonfinite position-contact accounting")
            result = {
                "profile": PROFILE,
                "steps": int(self.steps[None]),
                "columns": COLUMNS,
                "totals": dict(zip(COLUMNS, a.tolist(), strict=True)),
            }
            if reset:
                self.contact_totals.fill(0)
                self.interval.fill(0)
                self.steps.fill(0)
            return result

        def take_particle_contact(self, *, reset=True):
            data = self.interval.to_numpy()
            if reset:
                self.interval.fill(0)
            return {
                "columns": [
                    "contact_steps",
                    "normal_correction_m",
                    "tangent_dx_m",
                    "tangent_dy_m",
                    "tangent_dz_m",
                    "normal_impulse_kg_m_s",
                    "tangent_impulse_kg_m_s",
                    "friction_work_j",
                    "static_steps",
                ],
                "values": data,
            }

        @qd.func
        def _func_pbd_collide_with_rigid_geom(
            self,
            i,
            pos_world,
            vel,
            mass,
            normal_prev,
            geom_idx,
            batch_idx,
            geoms_state: array_class.GeomsState,
            geoms_info: array_class.GeomsInfo,
            links_state: array_class.LinksState,
            sdf_info: array_class.SDFInfo,
            rigid_info: array_class.RigidInfo,
            collider_static_config: qd.template(),
        ):
            new_pos, new_vel, normal = LegacyCoupler._func_pbd_collide_with_rigid_geom(
                self,
                i,
                pos_world,
                vel,
                mass,
                normal_prev,
                geom_idx,
                batch_idx,
                geoms_state,
                geoms_info,
                links_state,
                sdf_info,
                rigid_info,
                collider_static_config,
            )
            native_velocity = new_vel
            initial = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            dn = qd.max(0.0, (new_pos - pos_world).dot(normal))
            free = self.pbd_solver.particles_reordered[i, batch_idx].free
            top = normal[2] > 0.9999 and dn > 0.0
            delta = qd.Vector([0.0, 0.0, 0.0])
            stuck = False
            if free and top:
                displacement = new_pos - initial
                tangent = displacement - displacement.dot(normal) * normal
                length = tangent.norm()
                amount = qd.min(length, self.mu_kinetic * dn)
                if length <= self.mu_static * dn:
                    amount = length
                    stuck = True
                if length > 0:
                    delta = -tangent * (amount / length)
                new_pos += delta
                # Keep the native normal velocity exactly; delta is purely tangential.
                new_vel += delta / self.pbd_solver._substep_dt
                self.rigid_solver._func_apply_coupling_force(
                    geoms_info.link_idx[geom_idx],
                    batch_idx,
                    pos_world,
                    -mass * delta / (self.pbd_solver._substep_dt**2),
                    links_state,
                )
            dv = delta / self.pbd_solver._substep_dt
            work = mass * dv.dot(native_velocity) + 0.5 * mass * dv.dot(dv)
            jn = mass * dn / self.pbd_solver._substep_dt
            jt = mass * dv.norm()
            values = qd.Vector(
                [
                    dn,
                    delta.norm(),
                    jn,
                    jt,
                    work,
                    qd.cast(free and top, gs.qd_float),
                    qd.cast(free and top and stuck, gs.qd_float),
                    qd.cast(free and top and not stuck, gs.qd_float),
                    qd.cast(not free and delta.norm() > 0, gs.qd_float),
                    qd.max(0.0, work),
                ]
            )
            for col in qd.static(range(len(COLUMNS))):
                self.contact_totals[i, batch_idx, col] += qd.cast(values[col], qd.f64)
            # original ID is carried by the native reordered particle structure.
            original = self.original_id[i, batch_idx]
            interval = qd.Vector(
                [
                    qd.cast(free and top, gs.qd_float),
                    dn,
                    delta[0],
                    delta[1],
                    delta[2],
                    jn,
                    jt,
                    work,
                    qd.cast(free and top and stuck, gs.qd_float),
                ]
            )
            for col in qd.static(range(9)):
                self.interval[original, col] += qd.cast(interval[col], qd.f64)
            return new_pos, new_vel, normal

    result = PositionTableCoupler(scene.sim, scene.sim.coupler.options)
    scene.sim._coupler = result
    return result


def make_diagnostic_field(scene, n_particles):
    """Explicit external forces for coupons only; never installed in motion scenes."""
    import genesis as gs
    import quadrants as qd
    from genesis.engine.force_fields import ForceField

    @qd.data_oriented
    class MaskedLoad(ForceField):
        def __init__(self):
            super().__init__()
            self.mask = qd.field(gs.qd_bool, shape=n_particles)
            self.acceleration = qd.field(gs.qd_float, shape=())
            self.mask.fill(False)
            self.acceleration.fill(0)

        @qd.func
        def _get_acc(self, pos, vel, t, i):
            a = qd.Vector([0.0, 0.0, 0.0])
            if self.mask[i]:
                a[0] = self.acceleration[None]
            return a

    return scene.add_force_field(MaskedLoad())
