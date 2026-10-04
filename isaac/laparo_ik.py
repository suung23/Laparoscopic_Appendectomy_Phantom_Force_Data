#!/usr/bin/env python3
"""Laparoscopic RCM kinematics + strict-priority IK, engine-independent.

Extracted verbatim from `isaac/teleop_left_touch.py` so the teleop, the start-
pose aim (`isaac/aim_start_pose.py`) and the controller sweep
(`isaac/sweep_controller.py`) all run the SAME kinematics -- pure numpy, no
Isaac/Kit imports, so it can be imported before or without a SimulationApp.

Nothing here reads command-line ARGS: every gain is an explicit argument.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# URDF chain -> numpy FK / finite-difference Jacobian
# --------------------------------------------------------------------------
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def T_of(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


class Chain:
    """fr3_link0 -> tool_tip out of the URDF: 7 revolute + fixed tail."""

    def __init__(self, urdf_path: Path):
        import xml.etree.ElementTree as ET
        root = ET.parse(urdf_path).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        order = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
                 "fr3_joint5", "fr3_joint6", "fr3_joint7", "fr3_joint8",
                 "fr3_link8_to_tool", "tool_flange_to_tip"]
        self.steps = []          # (T_fixed_before, axis or None)
        self.lo, self.hi = [], []
        for name in order:
            j = joints[name]
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0")).split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0")).split()]
            T = T_of(xyz, rpy)
            if j.get("type") == "revolute":
                ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                lim = j.find("limit")
                self.lo.append(float(lim.get("lower")))
                self.hi.append(float(lim.get("upper")))
                self.steps.append((T, ax / np.linalg.norm(ax)))
            else:
                self.steps.append((T, None))
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)
        self.n = len(self.lo)
        # --- what fk() actually needs, hoisted out of the inner loop --------
        # fk is THE hot path of this whole project: one control tick is ~88
        # calls (3 IK iterations x 8 FD columns, plus 8 x 8 for the
        # manipulability gradient), and at 480 Hz that decides the sim's
        # real-time factor, which is what the operator feels as teleop lag.
        # Two things were being rebuilt on every call for every joint: the
        # skew matrix K and K @ K, which depend only on the AXIS and so are
        # constants; and a 4x4 homogeneous product where a 3x3 one plus a
        # matrix-vector does the identical arithmetic (the bottom row is
        # [0,0,0,1] and a joint step has no translation).
        self._Rf = [Tf[:3, :3].copy() for Tf, _ in self.steps]
        self._pf = [Tf[:3, 3].copy() for Tf, _ in self.steps]
        self._KK = []
        for _, ax in self.steps:
            if ax is None:
                self._KK.append(None)
                continue
            x, y, z = ax
            K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            self._KK.append((K, K @ K))
        self._I3 = np.eye(3)
        # fixed rotation flange -> tool TIP frame (the tail after fr3_joint8:
        # fr3_link8_to_tool then tool_flange_to_tip) -- lets callers reason
        # about the tip frame's axes (e.g. pointing its +y up) from R_flange
        T = np.eye(4)
        for Tf, ax in self.steps[8:]:
            T = T @ Tf
        self.R_f2t = T[:3, :3].copy()

    def fk(self, q, T_base=np.eye(4)):
        """Returns (p_tip, R_flange).

        Carried as a 3x3 R and a 3-vector p rather than a 4x4 T: identical
        arithmetic (a homogeneous product with a [0,0,0,1] bottom row is
        exactly `R @ Rf` and `R @ pf + p`) at a fraction of the allocation.
        See __init__ for the constants this leans on."""
        R = T_base[:3, :3]
        p = T_base[:3, 3]
        qi = 0
        R_flange = None
        for k in range(len(self.steps)):
            p = R @ self._pf[k] + p
            R = R @ self._Rf[k]
            kk = self._KK[k]
            if kk is not None:
                c, s = math.cos(q[qi]), math.sin(q[qi])
                R = R @ (self._I3 + s * kk[0] + (1.0 - c) * kk[1])
                qi += 1
            if k == 7:                       # after fr3_joint8: flange frame
                R_flange = R.copy()
        return p, R_flange

    def fk_points(self, q, T_base=np.eye(4)):
        """World positions along the arm: the origin after every chain step
        (joints 1..7, flange, tool joints, tip) -- consecutive pairs are the
        link segments used for the arm-vs-arm clearance check."""
        R = T_base[:3, :3]
        p = T_base[:3, 3]
        qi = 0
        pts = []
        for k in range(len(self.steps)):
            p = R @ self._pf[k] + p
            R = R @ self._Rf[k]
            pts.append(p)
            kk = self._KK[k]
            if kk is not None:
                c, s = math.cos(q[qi]), math.sin(q[qi])
                R = R @ (self._I3 + s * kk[0] + (1.0 - c) * kk[1])
                qi += 1
        return np.array(pts)                 # last point = tool tip


# shaft axis in the flange frame, unit, pointing flange -> tip. The motorised
# end-effector puts the tube COAXIAL with flange +Z (PCA of laparo_shaft.stl's
# 377 mm tube gives (1.1e-4, -2.8e-5, 1); the tip sits 0.03 mm off that line).
# It used to run sideways along flange -Y with the old hand-mounted tool.
SHAFT_AXIS_F = np.array([0.0, 0.0, 1.0])
# jaw clevis-pin axis in the flange frame (perpendicular to the shaft axis to
# 6e-5; see build_surgical_cell.py). Its world direction marks the tool's ROLL
# about the shaft -- the orientation DOF the RCM leaves free.
JAW_PIN_AXIS_F = np.array([0.707107, 0.707106, 0.0])


def perp_unit(v, a):
    """v projected perpendicular to unit axis a, normalised."""
    p = v - a * np.dot(v, a)
    n = np.linalg.norm(p)
    return p / n if n > 1e-9 else p


def roll_error(a, r_meas, r_cmd):
    """Signed angle (rad) to rotate r_meas onto r_cmd about the shaft axis a
    (both are projected perpendicular to a first)."""
    rm, rc = perp_unit(r_meas, a), perp_unit(r_cmd, a)
    return math.atan2(float(np.dot(a, np.cross(rm, rc))), float(np.dot(rm, rc)))


def rodrigues(a, ang):
    """Rotation matrix about unit axis a by ang."""
    x, y, z = a
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)

# trocar holes in the phantom STL frame (mm), 1-based list as measured
TROCAR_PORTS_MM = [
    (-95.171, 125.068, 133.195), (-53.106, 77.202, 175.961),
    (53.810, 79.279, 177.066), (98.208, 126.619, 134.031),
    (116.733, 16.543, 153.019), (-110.588, 10.071, 152.489),
    (56.075, -48.227, 180.953), (-50.545, -52.212, 179.154),
    (105.819, -109.028, 149.272), (-91.281, -113.699, 148.559),
    (8.161, -121.251, 164.292),
]


_NORMAL_CACHE: dict[int, np.ndarray] = {}


def port_inward_normal(port_no: int) -> np.ndarray:
    """Unit vector pointing INTO the phantom, perpendicular to the dome surface
    at the given trocar hole.

    Total-least-squares plane fit of the phantom.stl vertices within 15 mm of
    the hole (the fit rides over the hole's raised boss cleanly -- residuals
    are sub-mm -- where face-normal averaging is corrupted by the boss walls).
    Returned in the STL frame; the caller rotates it into the world by the pose
    the trocarPorts group carries.

    Memoised: the fit reloads a 19 MB STL, and the sweep asks for it per config.
    """
    if port_no in _NORMAL_CACHE:
        return _NORMAL_CACHE[port_no]
    import trimesh
    mesh = trimesh.load(ROOT / "phantom.stl", process=False)
    v = np.asarray(mesh.vertices, np.float64)
    p_mm = np.asarray(TROCAR_PORTS_MM[port_no - 1], np.float64)
    sel = v[np.linalg.norm(v - p_mm, axis=1) < 15.0]
    c = sel.mean(axis=0)
    _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
    nrm = vt[2] / np.linalg.norm(vt[2])            # least-variance direction
    interior = v.mean(axis=0)                      # dome centroid, well inside
    if np.dot(nrm, p_mm - interior) < 0:
        nrm = -nrm                                 # make it outward first
    _NORMAL_CACHE[port_no] = -nrm                  # inward
    return _NORMAL_CACHE[port_no]


def task_error(chain, q, T_base, p_cmd, p_rcm, w_rcm, r_cmd=None):
    p_tip, R_fl = chain.fk(q, T_base)
    a = R_fl @ SHAFT_AXIS_F                       # world shaft axis (toward tip)
    # closest point on the shaft line to the port centre
    p_c = p_tip + a * np.dot(p_rcm - p_tip, a)
    e = np.empty(6)
    e[:3] = p_cmd - p_tip
    e[3:] = w_rcm * (p_rcm - p_c)                 # perpendicular by construction
    # shaft-roll error (rad, about a): the one orientation DOF the RCM leaves
    # free. Handled OUTSIDE the primary task, in its null space (ik_step).
    phi = roll_error(a, R_fl @ JAW_PIN_AXIS_F, r_cmd) if r_cmd is not None else 0.0
    return e, phi, p_tip, a


def arm_jacobians(chain, q, T_base, p_cmd, p_rcm, w_rcm, r_cmd=None, h=1e-6):
    """Task error + finite-difference Jacobians at q.

    Returns (e0 (6), phi0, J (6 x n), Jr (n)). J is the true task Jacobian
    d(task)/dq: e = target - task, so (e0 - ei)/h = +d(task)/dq."""
    e0, phi0, _, _ = task_error(chain, q, T_base, p_cmd, p_rcm, w_rcm, r_cmd)
    J = np.empty((6, chain.n))
    Jr = np.empty(chain.n)
    for i in range(chain.n):
        qp = q.copy()
        qp[i] += h
        ei, phii, _, _ = task_error(chain, qp, T_base, p_cmd, p_rcm,
                                    w_rcm, r_cmd)
        J[:, i] = (e0 - ei) / h
        Jr[i] = (phi0 - phii) / h
    return e0, phi0, J, Jr


def rcm_manipulability(chain, q, T_base, p_rcm):
    """Yoshikawa measure of the TIP task RESTRICTED to the RCM null space:
    w = sqrt(det(Jc Jc^T)), Jc = J_tip (I - J_rcm^+ J_rcm).

    The plain 6-task determinant is useless here -- the RCM lateral rows have
    rank 2, so det = 0 identically. What actually matters for teleop feel is
    how well the tip can still move in EVERY direction while the shaft stays
    through the port; that is exactly this constrained measure. (p_cmd only
    shifts the error, not the Jacobian, so zeros(3) is fine.)

    N1 must be the FIXED-RANK-2 projector: the FD third singular direction of
    J_rcm is noise, and running it through a damped inverse made w jump ~70%
    over 1e-4 rad -- the 'gradient' was then noise that walked the null space
    straight into joint stops."""
    _, _, J, _ = arm_jacobians(chain, q, T_base, np.zeros(3), p_rcm, 1.0)
    N1 = np.eye(chain.n) - range_projector(J[3:], rank=2)
    Jc = J[:3] @ N1
    return math.sqrt(max(float(np.linalg.det(Jc @ Jc.T)), 0.0))


def manip_gradient(chain, q, T_base, p_rcm, k_manip=0.05):
    """The climb term of `posture_gradient`, on its own.

    Split out because it is the expensive half by a wide margin and the only
    half worth running at a lower rate: 8 finite-difference evaluations of
    `rcm_manipulability`, each of which is itself 8 forward kinematics, so 64
    of the ~88 FK calls a control tick makes. The other half (the joint-limit
    repulsion below) is analytic and effectively free, and must NOT be
    decimated -- it is what keeps the arm off its stops.

    Separately callable so the caller can hold it for several ticks; see
    ControlCfg.manip_hz for why that is safe and where it is not."""
    g = np.zeros(chain.n)
    if k_manip <= 0.0:
        return g
    w0 = rcm_manipulability(chain, q, T_base, p_rcm)
    if w0 <= 1e-12:
        return g
    h = 1e-4
    for i in range(chain.n):
        qp = q.copy()
        qp[i] += h
        g[i] = (rcm_manipulability(chain, qp, T_base, p_rcm) - w0) / h
    # d(log w)/dq, tamed: a log gradient explodes near w = 0 and would
    # otherwise drown the limit repulsion in FD noise
    return k_manip * np.clip(g / w0, -20.0, 20.0)


def posture_gradient(chain, q, T_base, p_rcm,
                     k_manip=0.05, k_limit=1.0, limit_margin=math.radians(15.0)):
    """Joint-space ascent direction for the posture (null-space) objective:
    log of the RCM-constrained manipulability (unit-free, so one gain fits all
    poses; clipped, because a log gradient explodes near w = 0 and would
    otherwise drown the limit repulsion in FD noise) plus a quadratic
    joint-limit repulsion that engages limit_margin from a stop. This is what
    keeps the ~2 redundant elbow DOF healthy instead of drifting into folds
    and stops."""
    g = manip_gradient(chain, q, T_base, p_rcm, k_manip)
    if k_limit > 0.0:
        d_lo = q - chain.lo
        d_hi = chain.hi - q
        g = g + k_limit * np.where(
            d_lo < limit_margin, ((limit_margin - d_lo) / limit_margin) ** 2, 0.0)
        g = g - k_limit * np.where(
            d_hi < limit_margin, ((limit_margin - d_hi) / limit_margin) ** 2, 0.0)
    return g


def elbow_up_gradient(chain, q, T_base, k=1.0, point=3, h=1e-6):
    """Null-space gradient that RAISES a chosen point on the arm in world z.

    The posture level otherwise optimises only constrained manipulability and
    joint-limit margin, and on this cell that settles with the elbow hanging low
    over the table -- the real rig carries it high, in an arch, which is both
    what the operator expects to see and what keeps the arm clear of the
    patient/table. This adds "lift point N" as an explicit preference so the
    start pose can be asked for it rather than seeded into it.

    `point` indexes chain.fk_points(): 3 is the joint-4 origin, the FR3's
    visual elbow. Returns dz/dq * k, to be fed to ik_step's g_extra."""
    p0 = chain.fk_points(q, T_base)[point]
    g = np.zeros(chain.n)
    for j in range(chain.n):
        qp = q.copy()
        qp[j] += h
        g[j] = (chain.fk_points(qp, T_base)[point][2] - p0[2]) / h
    return k * g


def range_projector(J, tol_rel=1e-2, rank=None):
    """V_r^T V_r: EXACT projector onto the row space of J (singular directions
    with sigma > tol_rel * sigma_max, or exactly the top `rank` when the task
    rank is known analytically). Used to build null-space restrictions that
    cannot leak: a damped pseudoinverse projector (I - J^T A^-1 J) is only
    approximate, and the leakage let the posture objective disturb tip/RCM
    tracking.

    Pass rank= for the RCM lateral task: it is rank 2 BY CONSTRUCTION (the
    lateral error lives perpendicular to the shaft), but its finite-difference
    third singular value is ~1e-5 noise, and any thresholded/damped treatment
    of that direction flickers frame to frame -- it made the manipulability
    measure (and its gradient) pure noise."""
    _, S, Vt = np.linalg.svd(J, full_matrices=False)
    if S.size == 0 or S[0] < 1e-12:
        return np.zeros((J.shape[1], J.shape[1]))
    Vr = Vt[:rank] if rank is not None else Vt[S > tol_rel * S[0]]
    return Vr.T @ Vr


def limit_scale(base, step, lo, hi, buf=math.radians(3.0)):
    """Largest s in [0,1] such that base + s*step keeps every joint out of the
    final `buf` of travel. Uniform, so the step direction is preserved. Used
    for the COMFORT levels (roll, posture): they must saturate at a stop, not
    ram it -- ramming hands the pose to the final clip, which bends the task
    direction. (The task levels are exempt: they may use the full range.)"""
    s = 1.0
    for i in range(len(step)):
        if step[i] > 1e-12:
            s = min(s, max(0.0, ((hi[i] - buf) - base[i]) / step[i]))
        elif step[i] < -1e-12:
            s = min(s, max(0.0, (base[i] - (lo[i] + buf)) / -step[i]))
    return s


def ik_step(chain, q, T_base, p_cmd, p_rcm, w_rcm, r_cmd=None,
            iters=3, damp=2e-3, damp_roll=5e-3, roll_clip=0.1,
            k_manip=0.05, k_limit=1.0, limit_margin=math.radians(15.0),
            g_extra=None, post_cap=0.01, roll_cap=0.02, null_acc=None):
    """STRICT-PRIORITY damped IK:

      L1  RCM lateral error               (scaled by w_rcm; fades -> released)
      L2  tip position, in null(L1)       (adaptive damping near singularity)
      L3  shaft roll,  in null(L1+L2)     (anti-windup-capped upstream)
      L4  posture,     in null(L1+L2+L3)  (manipulability + joint limits
                                           + g_extra, e.g. other-arm repulsion)

    The old single 6-task DLS let a tip command trade RCM error away and left
    the elbow self-motion unmanaged (it drifted into joint limits, where the
    final clip silently bent the commanded direction -- the 'moves the wrong
    way' symptom). Here the RCM always wins, the tip keeps its full 3 DOF
    inside the RCM null space, and the leftover redundancy is actively spent
    on staying manipulable, off the stops and off the other arm.

    The null-space restrictions are EXACT (SVD range projectors, blended out
    with w_rcm for the RCM level so a pulled-out tool is released smoothly);
    only the steps themselves are damped. With damped projectors the lower
    levels leak into the upper tasks and the posture objective visibly bends
    the tip path. The posture step is applied once per call (first iteration;
    the later iterations then re-converge the tasks on top of it), so its
    rate is bounded at post_cap per frame and it can never eat the whole
    qd_max budget."""
    n = chain.n
    I = np.eye(n)
    # posture direction once per call (it moves far slower than the task)
    gp = posture_gradient(chain, q, T_base, p_rcm, k_manip, k_limit,
                          limit_margin)
    if g_extra is not None:
        gp = gp + g_extra
    for it in range(iters):
        e0, phi0, J, Jr = arm_jacobians(chain, q, T_base, p_cmd, p_rcm,
                                        w_rcm, r_cmd)
        J_tip, J_rcm = J[:3], J[3:]
        # ---- L1: RCM ----
        A1 = J_rcm @ J_rcm.T + damp * np.eye(3)
        dq = J_rcm.T @ np.linalg.solve(A1, e0[3:])
        # exact null space, faded out with w_rcm (a projector is scale-
        # invariant, so the fade must be explicit: wb=1 locks the shaft onto
        # the port line, wb=0 frees the tip completely)
        wb = float(np.clip(w_rcm, 0.0, 1.0))
        N1 = I - wb * range_projector(J_rcm, rank=2)
        # ---- L2: tip, restricted to null(L1) ----
        J2 = J_tip @ N1
        sv = np.linalg.svd(J2, compute_uv=False)
        d2 = damp
        sig0 = 0.03                       # m/rad: constrained-singularity knee
        if sv[-1] < sig0:                 # boost damping as J2 degenerates so
            d2 = damp + 2e-2 * (1.0 - sv[-1] / sig0) ** 2   # poses degrade
        A2 = J2 @ J2.T + d2 * np.eye(3)                     # gracefully
        y2 = e0[:3] - J_tip @ dq
        dq = dq + J2.T @ np.linalg.solve(A2, y2)
        N2 = N1 @ (I - range_projector(J2))
        # ---- L3: shaft roll, restricted to null(L1+L2) ----
        if r_cmd is not None:
            jr2 = N2 @ Jr                 # safe direction: J_tip @ N2 = 0
            s = float(Jr @ jr2) + damp_roll
            dq3 = jr2 * ((np.clip(phi0, -roll_clip, roll_clip)
                          - float(Jr @ dq)) / s)
            # cap in JOINT space: where the null space can barely produce
            # roll, s -> damp_roll and the damped inverse amplifies --
            # uncapped, that spike derails the arm
            m3 = float(np.abs(dq3).max())
            if m3 > roll_cap:
                dq3 *= roll_cap / m3
            # and never let the ROLL level push a joint into the last few
            # degrees of travel: an unreachable roll must saturate (the
            # anti-windup upstream then holds the command near the achieved
            # roll), not ram a stop where the final clip bends the RCM/tip
            # direction. Uniform scale, so what roll survives stays pure roll.
            d3 = dq3 * limit_scale(q + dq, dq3, chain.lo, chain.hi)
            dq = dq + d3
            if null_acc is not None:
                null_acc += d3
            N3 = N2 @ (I - range_projector((Jr @ N2)[None, :]))
        else:
            N3 = N2
        # ---- L4: posture, in the remaining null space (first iter only) ----
        if it == 0:
            dq4 = N3 @ gp
            m4 = float(np.abs(dq4).max())
            if m4 > post_cap:
                dq4 *= post_cap / m4
            # same stop-buffer as the roll: a hard obstacle push (up to ~9x
            # the limit repulsion at full penetration) must not shove a joint
            # onto a stop and hand the pose to the clip
            d4 = dq4 * limit_scale(q + dq, dq4, chain.lo, chain.hi)
            dq = dq + d4
            if null_acc is not None:
                null_acc += d4
        # uniform scale (NOT per-joint clip: box-clipping distorts the dq
        # direction and re-couples the levels into each other)
        m = float(np.abs(dq).max())
        if m > 0.15:
            dq *= 0.15 / m
        q = np.clip(q + dq, chain.lo + 1e-3, chain.hi - 1e-3)
    return q


# --------------------------------------------------------------------------
# arm-vs-arm clearance: the arms are SEPARATE articulations, so PhysX lets a
# commanded pose ram the left arm into the right one and the drives stall
# fighting the contact -- the tool then veers instead of tracking. Repel in
# the posture level of the IK before it ever gets that far.
# --------------------------------------------------------------------------
def seg_seg_closest(p1, p2, p3, p4):
    """Closest points between segments [p1,p2] and [p3,p4]:
    returns (dist, c1, c2)."""
    d1, d2 = p2 - p1, p4 - p3
    r = p1 - p3
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r)), p1.copy(), p3.copy()
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    c1 = p1 + s * d1
    c2 = p3 + t * d2
    return float(np.linalg.norm(c1 - c2)), c1, c2


def obstacle_gradient(chain, q, T_base, obs_segs,
                      k_avoid=4.0, d_infl=0.10, r_sum=0.12):
    """Null-space repulsion from the OTHER arm.

    obs_segs: list of (a, b) world segments (the other arm's link axes).
    Left-arm segments come from fk_points on the COMMANDED q -- that is the
    pose the IK is about to ask for, which is what must stay clear. For every
    left segment whose surface distance to the closest obstacle is inside
    d_infl, push its material point away along the shortest line, mapped to
    joints with a finite-difference point Jacobian.

    Returns (g (n), clearance): g to feed ik_step's g_extra, clearance = the
    worst surface-to-surface distance (+inf if nothing is in range)."""
    pts0 = chain.fk_points(q, T_base)
    # skip the final segment (the tool SHAFT): under the RCM its line is fully
    # determined by tip + port -- both higher-priority tasks -- so a null-space
    # push can never move it and would only pollute the posture direction. The
    # wrist/flange ball that orbits during rolls (the real collision risk) is
    # the second-to-last segment and stays in.
    n_pts = len(pts0) - 1
    hits = []
    clearance = float("inf")
    for i in range(n_pts - 1):
        best = None
        for (oa, ob) in obs_segs:
            d, c1, _c2 = seg_seg_closest(pts0[i], pts0[i + 1], oa, ob)
            if best is None or d < best[0]:
                best = (d, c1, _c2)
        if best is None:
            continue
        d, c1, c2 = best
        surf = d - r_sum
        clearance = min(clearance, surf)
        if surf < d_infl and d > 1e-9:
            # material point on the left segment, its fraction, push direction
            s = (float(np.linalg.norm(c1 - pts0[i]))
                 / max(float(np.linalg.norm(pts0[i + 1] - pts0[i])), 1e-9))
            u = (c1 - c2) / d
            mag = k_avoid * min((d_infl - surf) / d_infl, 1.5) ** 2
            hits.append((i, np.clip(s, 0.0, 1.0), u * mag))
    g = np.zeros(chain.n)
    if hits:
        # one FD sweep gives every point Jacobian at once
        h = 1e-6
        dpts = []
        for j in range(chain.n):
            qp = q.copy()
            qp[j] += h
            dpts.append((chain.fk_points(qp, T_base) - pts0) / h)
        for (i, s, f) in hits:
            for j in range(chain.n):
                jp = (1.0 - s) * dpts[j][i] + s * dpts[j][i + 1]
                g[j] += float(jp @ f)
    return g, clearance
