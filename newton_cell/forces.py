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
    """Per-arm 6D wrench from the soft contacts on that arm's bodies."""

    def __init__(self, model, arms):
        self.model = model
        self.arms = arms
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

    def read(self, state, contacts) -> dict:
        """(force, torque) in WORLD, per arm, from the current contact set."""
        out = {k: (np.zeros(3), np.zeros(3)) for k in self.arms}
        n = int(contacts.soft_contact_count.numpy()[0])
        if n <= 0:
            return out
        n = min(n, int(contacts.soft_contact_max))
        part = contacts.soft_contact_particle.numpy()[:n]
        shape = contacts.soft_contact_shape.numpy()[:n]
        normal = contacts.soft_contact_normal.numpy()[:n]
        bpos = contacts.soft_contact_body_pos.numpy()[:n]
        bvel = contacts.soft_contact_body_vel.numpy()[:n]
        q = state.particle_q.numpy()
        qd = state.particle_qd.numpy()
        bq = state.body_q.numpy() if self.model.body_count else None

        ke = float(self.model.soft_contact_ke)
        kd = float(self.model.soft_contact_kd)
        for i in range(n):
            sh = int(shape[i])
            if sh < 0 or self._shape_body is None:
                continue
            body = int(self._shape_body[sh])
            arm = self.body_arm.get(body)
            if arm is None:
                continue
            p = int(part[i])
            nrm = normal[i]
            # contact point on the body, in world
            cp = bpos[i]
            if bq is not None:
                t = bq[body]
                R = _quat_to_R(t[3:7])
                cp = R @ cp + t[0:3]
            # the same penalty Newton integrates
            d = float(np.dot(q[p] - cp, nrm))
            pen = self.radius - d
            if pen <= 0.0:
                continue
            vrel = qd[p] - bvel[i]
            fmag = ke * pen - kd * float(np.dot(vrel, nrm))
            if fmag <= 0.0:
                continue
            f = fmag * nrm
            fo, to = out[arm]
            # sign: the load the TISSUE puts on the INSTRUMENT
            out[arm] = (fo - f, to - np.cross(cp, f))
        return out

    def tool_wrench(self, state, contacts, arm_name) -> np.ndarray:
        """The 6D wrench on `arm_name`, in that arm's tool_tip frame."""
        f_w, t_w = self.read(state, contacts)[arm_name]
        arm = self.arms[arm_name]
        T = arm.link_frames()["tool_tip"]
        R, p = T[:3, :3], T[:3, 3]
        # move the torque to the tip, then rotate both into the tip frame
        t_tip = t_w + np.cross(p, f_w)
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
