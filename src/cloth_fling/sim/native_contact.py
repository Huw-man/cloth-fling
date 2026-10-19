"""Scene-local Genesis 0.4.3 normal-impulse Coulomb extension.

No Genesis or torch imports until installation. Never edits upstream files or classes.
"""

import ast
import hashlib
import inspect
import textwrap

CONTACT_TRACE_COLUMNS = (
    "normal_impulse_kg_m_s",
    "tangential_impulse_kg_m_s",
    "friction_work_j",
    "native_normal_work_j",
    "free_top_contacts",
    "attached_top_contacts",
    "noncontact_calls",
    "attached_friction_events",
    "free_friction_events",
    "eligible_normal_impulse_kg_m_s",
    "noncontact_friction_events",
    "positive_friction_work_j",
)
CONTACT_TRACE_CAPACITY = 8192

_SOURCE_HASHES = {
    "_func_pbd_collide_with_rigid_geom": "2fa83fee4c1435d6dfa68118e049c31974167c2147067b8a19fb90a66dfd62e4",
    "kernel_pbd_rigid_collide": "06805ee8f1a3068791d22c08bbdd80cd4024095f1ed553f17eb97d03b3d7faa5",
}


def validate_native_source(version, source):
    """AST guard ignores whitespace/comments, rejects changed executable source."""
    if version != "0.4.3":
        raise RuntimeError("native_coulomb_v1 requires Genesis 0.4.3")
    try:
        tree = ast.parse(textwrap.dedent(source))
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LegacyCoupler"
        )
        actual = {
            n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in _SOURCE_HASHES
        }
    except (SyntaxError, StopIteration) as error:
        raise RuntimeError("unsupported Genesis native contact source") from error
    if actual != _SOURCE_HASHES:
        raise RuntimeError(f"unsupported Genesis 0.4.3 native contact source: {actual}")
    return actual


def install_native_contact(scene, table, config):
    """Install an isolated subclass instance before build; fixed single table only.

    Uses the native normal response unchanged. Tangential impulse is bounded by
    mu times mass * max(0, (v_native - v_in).n), including landing impulses.
    No mg proxy, normal damping, position friction or poststep setters.
    """
    import genesis as gs
    import quadrants as qd
    from genesis.engine.couplers.legacy_coupler import LegacyCoupler
    from genesis.utils import array_class

    validate_native_source(gs.__version__, inspect.getsource(LegacyCoupler))
    if scene.is_built:
        raise RuntimeError("native contact must be installed before scene.build")
    if (
        type(scene.sim.coupler) is not LegacyCoupler
        or len(scene.rigid_solver.entities) != 1
        or scene.rigid_solver.entities[0] is not table
        or not table.morph.fixed
        or type(table.morph).__name__ != "Box"
        or len(table.geoms) != 1
    ):
        raise RuntimeError("native contact requires one fixed Box table and a legacy coupler")
    static, kinetic = config.table_coefficients if config.table_friction else (0.0, 0.0)

    @qd.data_oriented
    class NativeTableCoupler(LegacyCoupler):
        def __init__(self, simulator, options):
            super().__init__(simulator, options)
            self.table_mu_static = float(static)
            self.table_mu_kinetic = float(kinetic)

        def build(self):
            super().build()
            self.contact_trace = qd.field(
                gs.qd_float, shape=(CONTACT_TRACE_CAPACITY, len(CONTACT_TRACE_COLUMNS))
            )
            self.contact_step = qd.field(gs.qd_int, shape=())
            self.capture_enabled = qd.field(gs.qd_int, shape=())
            self.accounting_steps = qd.field(qd.i64, shape=())
            # Per-particle float64 totals avoid hot global atomics and loss of
            # integer participation precision during long/repeated actions.
            self.contact_totals = qd.field(
                qd.f64,
                shape=(
                    self.pbd_solver._n_particles,
                    self.pbd_solver._B,
                    len(CONTACT_TRACE_COLUMNS),
                ),
            )
            self.contact_trace.fill(0)
            self.contact_step.fill(-1)
            self.capture_enabled.fill(0)
            self._accounting_epoch = 0
            self.contact_totals.fill(0)
            self.accounting_steps.fill(0)

        def take_contact_accounting(self, *, reset=False):
            """Read cumulative totals on demand; scene resets never clear them.

            reset=True returns the previous epoch then clears cumulative totals.
            Bounded diagnostic capture is independent and explicitly managed.
            """
            import numpy as np

            values = self.contact_totals.to_numpy().sum(axis=(0, 1))
            if not np.isfinite(values).all():
                raise RuntimeError("nonfinite native contact accounting")
            result = {
                "schema_version": 1,
                "profile": "native_coulomb_v1",
                "epoch": self._accounting_epoch,
                "substeps": int(self.accounting_steps[None]),
                "totals": dict(zip(CONTACT_TRACE_COLUMNS, values.tolist(), strict=True)),
            }
            if reset:
                self.contact_totals.fill(0)
                self.accounting_steps.fill(0)
                self._accounting_epoch += 1
            return result

        def clear_contact_trace(self):
            """Opt in to a fresh bounded substep capture; no cumulative reset."""
            self.contact_trace.fill(0)
            self.contact_step.fill(-1)
            self.capture_enabled.fill(1)

        def stop_contact_trace(self):
            """Disable capture without clearing its data or cumulative totals."""
            self.capture_enabled.fill(0)

        def read_contact_trace(self):
            """One device readback per scenario, with explicit overflow detection."""
            steps = int(self.contact_step[None]) + 1
            return {
                "columns": CONTACT_TRACE_COLUMNS,
                "substeps": steps,
                "truncated": steps > CONTACT_TRACE_CAPACITY,
                "values": self.contact_trace.to_numpy()[: min(steps, CONTACT_TRACE_CAPACITY)],
            }

        @qd.kernel
        def _begin_contact_substep(self):
            self.accounting_steps[None] += 1
            if self.capture_enabled[None]:
                self.contact_step[None] += 1

        def couple(self, f):
            self._begin_contact_substep()
            super().couple(f)

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
            rigid_global_info: array_class.RigidGlobalInfo,
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
                rigid_global_info,
                collider_static_config,
            )
            normal_velocity = new_vel
            normal_dv = qd.max(0.0, (new_vel - vel).dot(normal))
            free = self.pbd_solver.particles_reordered[i, batch_idx].free
            removed = 0.0
            friction_dv = qd.Vector([0.0, 0.0, 0.0])
            # Restrict friction to the horizontal top; sides keep native response.
            if free and normal[2] > 0.9999:
                tangent = new_vel - new_vel.dot(normal) * normal
                speed = tangent.norm()
                if normal_dv > 0.0 and speed > 0.0:
                    removed = qd.min(speed, self.table_mu_kinetic * normal_dv)
                    if speed <= self.table_mu_static * normal_dv:
                        removed = speed
                    friction_dv = -tangent * (removed / speed)
                    new_vel += friction_dv
                    # Fixed table: retain equal/opposite native coupling accounting.
                    self.rigid_solver._func_apply_coupling_force(
                        pos_world,
                        -mass * friction_dv / self.rigid_solver._substep_dt,
                        geoms_info.link_idx[geom_idx],
                        batch_idx,
                        links_state,
                    )
            # Accounting never participates in the physical impulse calculation.
            work = mass * friction_dv.dot(normal_velocity) + 0.5 * mass * friction_dv.dot(
                friction_dv
            )
            normal_delta = normal_velocity - vel
            normal_work = mass * normal_delta.dot(vel) + 0.5 * mass * normal_delta.dot(normal_delta)
            top_contact = normal_dv > 0.0 and normal[2] > 0.9999
            eligible_normal_impulse = 0.0
            if top_contact and free:
                eligible_normal_impulse = mass * normal_dv
            values = qd.Vector(
                [
                    mass * normal_dv,
                    mass * removed,
                    work,
                    normal_work,
                    qd.cast(top_contact and free, gs.qd_float),
                    qd.cast(top_contact and not free, gs.qd_float),
                    qd.cast(normal_dv <= 0.0, gs.qd_float),
                    qd.cast(removed > 0.0 and not free, gs.qd_float),
                    qd.cast(removed > 0.0 and free, gs.qd_float),
                    eligible_normal_impulse,
                    qd.cast(removed > 0.0 and normal_dv <= 0.0, gs.qd_float),
                    qd.max(0.0, work),
                ]
            )
            for column in qd.static(range(len(CONTACT_TRACE_COLUMNS))):
                self.contact_totals[i, batch_idx, column] += qd.cast(values[column], qd.f64)
                if self.capture_enabled[None]:
                    row = qd.max(self.contact_step[None], 0)
                    if row < CONTACT_TRACE_CAPACITY:
                        self.contact_trace[row, column] += values[column]
            return new_pos, new_vel, normal

    coupler = NativeTableCoupler(scene.sim, scene.sim.coupler.options)
    scene.sim._coupler = coupler
    return coupler
