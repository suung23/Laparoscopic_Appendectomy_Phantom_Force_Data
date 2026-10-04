"""Engine-independent adapter to Rosota's original FR3 controller.

Requires a real FR3/tool Chain and measured joint feedback. This module does
not turn the legacy CT scene's kinematic tool bodies into FR3 articulations.
The strict aperture profile is an explicit override of upstream's 60-degree
exploratory cap; the original controller and IK files remain unchanged.
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaac'))
import trocar
from laparo_control import BEST, TipController
from laparo_ik import JAW_PIN_AXIS_F, perp_unit, rodrigues, roll_error

STRICT_CFG = BEST.replace(name='rosota-best-strict-aperture', tilt_cap_deg=0.0, manip_hz=60.0)


def aperture_metrics(port, normal, tip, inward_axis):
    vectors = [np.asarray(v, float) for v in (port, normal, tip, inward_axis)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError('Finite 3D port, normal, tip and shaft axis required')
    p, n, t, a = vectors
    if min(np.linalg.norm(n), np.linalg.norm(a)) < 1e-12:
        raise ValueError('Nonzero normal and shaft axis required')
    n = n / np.linalg.norm(n); a = a / np.linalg.norm(a)
    raw = trocar.port_geometry(p, n, t, a)
    insertion = float(np.dot(t-p, a))
    inward = bool(np.dot(a, n) > 0 and insertion > 0)
    return dict(offset_m=float(raw['off'][0]), bite_m=float(raw['bite'][0]),
                tilt_rad=float(raw['tilt'][0]), insertion_m=insertion,
                inward=inward, aperture_clear=bool(inward and raw['bite'][0] <= 1e-9))


def check_chunk(action, max_chunk=8):
    """Strict array subset of Rosota (2,K,7); validate both arms atomically."""
    a = np.asarray(action, float)
    if a.ndim != 3 or a.shape[0] != 2 or a.shape[2] != 7 or not 1 <= a.shape[1] <= max_chunk:
        raise ValueError('Expected (2,K,7), 1 <= K <= max_chunk')
    if not np.isfinite(a).all() or np.any((a[:,:,6] < 0) | (a[:,:,6] > 100)):
        raise ValueError('Finite action and absolute gripper in [0,100] required')
    return a.copy()


class RosotaArmController:
    """Use at 480 Hz with actual measured q; outputs joint targets and feedforward.

    Tool deltas use the URDF tool_tip frame: -z inserts, +rz is opposite
    FR3 joint 7. Call begin_substep once per 30 Hz action sample, then tick
    sixteen times. Each tick must be followed by a plant step and feedback.
    Jaw output is an included-angle command; force-limited closure is the
    plant's responsibility and is not fabricated here.
    """
    def __init__(self, chain, base, port, normal, q0, *, hand):
        if hand not in ('left', 'right'): raise ValueError('Unknown hand')
        self.chain = chain; self.base = np.asarray(base, float).copy()
        if self.base.shape != (4,4) or not np.isfinite(self.base).all() or not np.allclose(self.base[3], [0,0,0,1]) or not np.allclose(self.base[:3,:3].T @ self.base[:3,:3], np.eye(3)) or not np.isclose(np.linalg.det(self.base[:3,:3]), 1):
            raise ValueError('Rigid base transform required')
        self.port = np.asarray(port, float).copy()
        self.normal = np.asarray(normal, float).copy()
        q0 = self._q(q0)
        p, R, axis = self.pose(q0)
        initial = aperture_metrics(self.port, self.normal, p, axis)
        if not initial['aperture_clear'] or initial['insertion_m'] < .1-1e-9:
            raise ValueError('Initial measured pose violates port or 100 mm insertion floor')
        self.normal /= np.linalg.norm(self.normal)
        self.ctl = TipController(STRICT_CFG, chain, self.base, self.port, q0,
                                 min_depth=.1, port_normal=self.normal)
        self.p_cmd = p.copy(); self.v_cmd = np.zeros(3)
        self.r_cmd = perp_unit(R[:,1], axis)
        self.roll_cmd = self.roll_put = 0.0
        self.open_deg = 74.0 if hand == 'left' else 50.0
        self.remaining = 0

    def _q(self, q):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.isfinite(q).all() or np.any(q < self.chain.lo) or np.any(q > self.chain.hi):
            raise ValueError('Seven measured FR3 joint positions within limits required')
        return q.copy()

    def pose(self, q):
        p, flange = self.chain.fk(q, self.base)
        R = flange @ self.chain.R_f2t
        return p, R, -R[:,2]

    def begin_substep(self, row, q_measured):
        row = np.asarray(row, float)
        q = self._q(q_measured)
        if row.shape != (7,) or not np.isfinite(row).all() or not 0 <= row[6] <= 100:
            raise ValueError('Finite seven-component Rosota action required')
        if self.remaining: raise RuntimeError('Finish pending physics ticks first')
        self.step_frame = self.pose(q)[1].copy()
        self.row = row.copy(); self.remaining = 16

    def tick(self, q_measured):
        q = self._q(q_measured)
        if not self.remaining: raise RuntimeError('begin_substep required')
        dt = 1/480
        p, R, axis = self.pose(q)
        measured = aperture_metrics(self.port, self.normal, p, axis)
        if not measured['aperture_clear'] or measured['insertion_m'] < .1-1e-6:
            raise RuntimeError('Measured trocar aperture/depth violation; stop episode')
        old = self.p_cmd.copy()
        cmd = old + self.step_frame @ (self.row[:3]/16)
        v = cmd-self.port; depth = np.linalg.norm(v)
        if depth < .1:
            cmd = self.port + (v/depth if depth > 1e-12 else self.normal)*.1
        cmd, cone_hit = trocar.cone_clamp(cmd, self.port, self.normal, self.ctl.tilt_max)
        delta = cmd-p; distance = np.linalg.norm(delta)
        leash_hit = distance > .015
        if leash_hit: cmd = p+delta*(.015/distance)
        self.p_cmd = cmd
        self.v_cmd += ((cmd-old)/dt-self.v_cmd)*(1-math.exp(-2*math.pi*8*dt))
        droll = float(self.row[5]/16)
        self.roll_cmd += droll
        j = self.ctl.ROLL_JOINT
        residual = -(self.roll_cmd-self.roll_put)
        residual = float(np.clip(residual, min(self.chain.lo[j]+.001-self.ctl.q[j],0),
                                max(self.chain.hi[j]-.001-self.ctl.q[j],0)))
        self.roll_cmd = self.roll_put-residual
        self.r_cmd = perp_unit(rodrigues(axis, -droll) @ self.r_cmd, axis)
        flange = R @ self.chain.R_f2t.T
        phi = roll_error(axis, flange @ JAW_PIN_AXIS_F, self.r_cmd)
        cap = math.radians(45)
        if abs(phi)>cap:
            self.r_cmd = perp_unit(rodrigues(axis, math.copysign(cap,phi)-phi) @ self.r_cmd, axis)
        before = float(self.ctl.q[j])
        target, velocity, shaped = self.ctl.step(cmd, self.v_cmd, p, dt,
            r_cmd=self.r_cmd, line_meas=(p,axis), q_meas=q, droll=residual)
        self.roll_put -= float(self.ctl.q[j]-before)
        self.remaining -= 1
        return dict(joint_target=target.copy(), joint_velocity_ff=velocity.copy(),
            shaped_tip=shaped.copy(), gripper_target_deg=float(self.row[6]/100*self.open_deg),
            gripper_rate_dps=250., rot_rejected=self.row[3:5].copy()/16,
            roll_backlog_rad=self.roll_cmd-self.roll_put,
            cone_clamped=bool(cone_hit), leash_clamped=bool(leash_hit), trocar=measured)
