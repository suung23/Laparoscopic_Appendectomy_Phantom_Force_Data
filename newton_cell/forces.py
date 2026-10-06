#!/usr/bin/env python3
"""Instrument contact force, read off the tissue instead of a joint sensor.

The Isaac backend reads a PhysX articulation reaction sensor at the tool mount
and then has to work very hard to make it usable: the instrument's own 4.7 N
weight swamps the tens of millinewtons of tissue contact, and subtracting it
analytically from the USD mass/COM left a 4.2 N residual that DRIFTED 0.8 N over
a 19-degree roll. VLA_HANDOFF.md section 4.2 describes the fix -- identify a
wrench model over 54 static poses, cache it, and tare at reset.

None of that applies here. The instrument is kinematically driven, so it has no
joint reaction to read at all; instead the force is taken directly from the
soft-body contacts the tissue makes with the instrument. That measures the thing
the channel is actually for, and it removes the gravity term entirely rather
than identifying and subtracting it -- there is no 4.7 N to cancel, so there is
no 311 mN identification residual either.

The force uses the same penalty model Newton integrates
(`f = ke * penetration` along the contact normal, plus the velocity damping
term), so it is the force the solver actually applied, not an approximation of
a different model.

Reported on the URDF `tool_tip` axes, exactly as the handoff defines them:

    bend_x   tip +x   the jaw closing direction
    bend_y   tip +y   the clevis pin axis
    axial    tip -z   the INSERTION direction
    torsion  tip -z   about the insertion axis

and signed as the load the TISSUE applies to the INSTRUMENT, so pushing the tip
into an organ gives `axial < 0`.
"""
from __future__ import annotations

import numpy as np


class ContactForces:
    """Reconstructed per-arm wrench from Newton's unified soft contacts.

    This is a kinematic-body reaction estimate, not a sensor readback. Newton's
    unified stream contains vertex, edge, and face records; its legacy
    ``soft_contact_particle`` entry is -1 for edge/face records. Never use that
    sentinel as a Python index: it aliases the final tissue vertex.
    """

    def __init__(self, model, arms, solver=None, dt=1.0 / 480.0):
        self.model = model
        self.arms = arms
        self.solver = solver
        self.dt = float(dt)
        # shape index -> arm name, for attributing each contact
        self.shape_arm = {}
        for name, arm in arms.items():
            for link, body in arm.bodies.items():
                for s in range(model.shape_count):
                    pass
        # body -> arm is enough: soft_contact_shape gives a shape, and
        # model.shape_body maps it to the body we created per link.
        self.body_arm = {}
        for name, arm in arms.items():
            for link, body in arm.bodies.items():
                self.body_arm[int(body)] = name
        self._shape_body = model.shape_body.numpy() if model.shape_count else None
        self.radius = float(np.max(model.particle_radius.numpy())) \
            if model.particle_count else 0.0

    @staticmethod
    def _numpy(value):
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    def _material(self, index, shape):
        """Use static mixed stiffness and VBD's cached damping/friction.

        VBD's ``body_particle_contact_penalty_k`` is a dual-updated solve
        variable. Reading it after ``solver.step`` does not recover the
        coefficient used for the contact forces inside that step; treating it
        as a sensor gain generated multi-meganewton post-solve artifacts.
        """
        ke = (float(self.model.soft_contact_ke) +
              float(self._numpy(self.model.shape_material_ke)[shape])) * 0.5
        if self.solver is not None:
            names = ("body_particle_contact_material_kd",
                     "body_particle_contact_material_mu")
            arrays = [getattr(self.solver, name, None) for name in names]
            if all(a is not None and int(a.shape[0]) > index for a in arrays):
                kd, mu = tuple(float(self._numpy(a)[index]) for a in arrays)
                return ke, kd, mu
        kd = (float(self.model.soft_contact_kd) +
              float(self._numpy(self.model.shape_material_kd)[shape])) * 0.5
        mu = (float(self.model.soft_contact_mu) +
              float(self._numpy(self.model.shape_material_mu)[shape])) * 0.5
        return ke, kd, mu

    def read(self, state, contacts) -> dict:
        """(force, torque) in WORLD, per arm, from the current contact set."""
        out = {k: (np.zeros(3), np.zeros(3)) for k in self.arms}
        n = int(contacts.soft_contact_count.numpy()[0])
        if n <= 0:
            return out
        n = min(n, int(contacts.soft_contact_max))
        indices = self._numpy(contacts.soft_contact_indices)[:n]
        bary = self._numpy(contacts.soft_contact_barycentric)[:n]
        shape = contacts.soft_contact_shape.numpy()[:n]
        normal = contacts.soft_contact_normal.numpy()[:n]
        bpos = contacts.soft_contact_body_pos.numpy()[:n]
        bvel = contacts.soft_contact_body_vel.numpy()[:n]
        q = self._numpy(state.particle_q)
        qd = self._numpy(state.particle_qd)
        bq = state.body_q.numpy() if self.model.body_count else None
        q_prev = q
        body_q_prev = bq
        if self.solver is not None:
            qp = getattr(self.solver, "particle_q_prev", None)
            bp = getattr(self.solver, "body_q_prev", None)
            if qp is not None and int(qp.shape[0]) == len(q):
                q_prev = self._numpy(qp)
            if bp is not None and bq is not None and int(bp.shape[0]) == len(bq):
                body_q_prev = self._numpy(bp)
        radii = self._numpy(self.model.particle_radius)
        margins = self._numpy(self.model.shape_margin)
        dt = self.dt
        friction_epsilon = float(getattr(self.solver, "friction_epsilon", 1.0e-3))
        for i in range(n):
            sh = int(shape[i])
            if sh < 0 or sh >= len(self._shape_body):
                continue
            body = int(self._shape_body[sh])
            arm = self.body_arm.get(body)
            if arm is None:
                continue
            nrm = np.asarray(normal[i], dtype=np.float64)
            norm = float(np.linalg.norm(nrm))
            if not np.isfinite(nrm).all() or norm < 1e-9:
                continue
            nrm /= norm
            nodes = np.asarray(indices[i], dtype=np.int64)
            valid = (nodes >= 0) & (nodes < len(q))
            if not valid.any():
                continue
            nodes = nodes[valid]
            weights = np.asarray(bary[i][valid], dtype=np.float64)
            if not np.isfinite(weights).all() or weights.sum() <= 1e-9:
                continue
            weights /= weights.sum()
            point = np.sum(q[nodes] * weights[:, None], axis=0)
            point_prev = np.sum(q_prev[nodes] * weights[:, None], axis=0)
            cp_local = np.asarray(bpos[i], dtype=np.float64)
            cp, cp_prev = cp_local.copy(), cp_local.copy()
            body_R = np.eye(3)
            if bq is not None:
                if body < 0 or body >= len(bq):
                    continue
                t = bq[body]
                tp = body_q_prev[body]
                body_R = _quat_to_R(t[3:7])
                body_R_prev = _quat_to_R(tp[3:7])
                cp = body_R @ cp_local + t[0:3]
                cp_prev = body_R_prev @ cp_local + tp[0:3]
            # Newton stores this surface velocity in body-local coordinates.
            body_v = body_R @ bvel[i]
            relative = (point - point_prev) - (cp - cp_prev) - body_v * dt
            depth = float(np.max(radii[nodes]) + margins[sh] -
                          np.dot(nrm, point - cp))
            if not np.isfinite(depth) or depth <= 0.0:
                continue
            ke, kd, mu = self._material(i, sh)
            elastic_load = ke * depth
            normal_rate = float(np.dot(nrm, relative))
            normal_load = elastic_load - (kd / dt) * min(normal_rate, 0.0)
            tangent = relative - nrm * normal_rate
            tangent_len = float(np.linalg.norm(tangent))
            friction = np.zeros(3)
            eps = friction_epsilon * dt
            if mu > 0.0 and elastic_load > 0.0 and tangent_len > 0.0 and eps > 0.0:
                scale = (1.0 / tangent_len if tangent_len > eps else
                         (-tangent_len / eps + 2.0) / eps)
                friction = -(mu * elastic_load * scale) * tangent
            f_body = -(normal_load * nrm + friction)
            if not np.isfinite(f_body).all():
                continue
            fo, to = out[arm]
            out[arm] = (fo + f_body, to + np.cross(cp, f_body))
        return out

    def tool_wrench(self, state, contacts, arm_name) -> np.ndarray:
        """The 6D wrench on `arm_name`, in that arm's tool_tip frame."""
        f_w, t_w = self.read(state, contacts)[arm_name]
        arm = self.arms[arm_name]
        T = arm.link_frames()["tool_tip"]
        R, p = T[:3, :3], T[:3, 3]
        # move the torque to the tip, then rotate both into the tip frame
        t_tip = t_w - np.cross(p, f_w)
        return np.concatenate([R.T @ f_w, R.T @ t_tip])

    @staticmethod
    def channels(wrench) -> dict:
        """The named channels of VLA_HANDOFF.md section 4.2 from a tool wrench."""
        return dict(
            bend_x=float(wrench[0]),
            bend_y=float(wrench[1]),
            axial=float(-wrench[2]),      # tip -z is the insertion direction
            torsion=float(-wrench[5]),
            wrench_tool=np.asarray(wrench, np.float32),
        )


def _quat_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
