#!/usr/bin/env python3
"""The tool-tip controller, as a parameterised family so it can be swept.

Everything between "here is where the operator wants the tip" and "here is what
PhysX is asked to do" lives here, with every knob that the literature (and the
existing teleop) offers exposed as one field of `ControlCfg`:

    command shaping   lead time, low-pass, admittance relaxation
    redundancy        hard (strict-priority null-space) vs soft (weighted DLS)
                      remote-centre-of-motion constraint, IK damping, iterations
    joint servo       drive stiffness kp, damping kd, velocity feed-forward,
                      commanded joint-speed clamp

`isaac/sweep_controller.py` grid-searches this; the winning row becomes
`BEST`, which `isaac/teleop_left_touch.py` uses as its defaults. Pure numpy
except for the ArticulationAction it hands back, so it can be unit-tested and
replayed off-engine.
"""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trocar  # noqa: E402
from laparo_ik import (arm_jacobians, elbow_up_gradient,  # noqa: E402
                       limit_scale, manip_gradient, posture_gradient,
                       range_projector)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
GUARD_OFF = -99.0     # bite_guard_mm sentinel: the guard is disabled. Not
                      # "any negative", because a negative THRESHOLD is the
                      # useful setting -- it demands clearance, not just
                      # absence of contact.


@dataclass
class ControlCfg:
    """One point in the controller design space."""

    # ---- redundancy resolution / RCM ----
    rcm_mode: str = "hard"        # "hard" strict-priority | "soft" weighted DLS
    rcm_weight: float = 1.0       # soft only: weight on the RCM rows
    damp: float = 2e-3            # DLS damping (all levels)
    iters: int = 3                # IK iterations per control tick
    k_manip: float = 0.05         # posture: constrained-manipulability climb
    manip_hz: float = 0.0         # Hz to recompute that climb at (0 = every
                                  # tick). It is 64 of the ~88 forward
                                  # kinematics a tick costs -- 8 finite
                                  # differences of a measure that is itself 8
                                  # more -- and it steers the ELBOW, which
                                  # moves in fractions of a degree per second.
                                  # Held between refreshes, it costs nothing
                                  # the objective can notice: measured, the
                                  # term's own FD noise floor (1.3e-3 on a
                                  # +-1.0 clip) is larger than the drift a
                                  # 60 Hz hold introduces. The joint-stop
                                  # repulsion is NOT decimated with it -- see
                                  # laparo_ik.manip_gradient
    k_limit: float = 1.0          # posture: joint-stop repulsion
    limit_margin_deg: float = 15.0

    # ---- the port as a PHYSICAL aperture (isaac/trocar.py) ----
    rcm_fb: float = 0.0           # gain on the MEASURED shaft-vs-port error
    rcm_fb_max: float = 0.005     # m: cap on the resulting target bias
    rcm_fb_lpf_hz: float = 0.0    # low-pass on the measured error, 0 = off
    clamp_mode: str = "joint"     # joint-speed clamp: "joint"|"task"|"null"
    post_cap: float = 0.01        # rad: cap on the posture step, per call
    roll_cap: float = 0.02        # rad: cap on the roll step, per iteration
                                  # Both are NULL-SPACE comfort objectives, and
                                  # both are bigger than the whole per-tick
                                  # joint budget (qd_max/480 = 4.2 mrad) -- so
                                  # they are what trips the speed clamp, and
                                  # the clamp then shrinks the RCM rows with
                                  # them. See the `port` stage.
    cone_mm: float = -1.0         # mm of lateral room the tilt cone reserves
                                  # at the aperture; negative = no cone clamp
    tilt_cap_deg: float = trocar.TILT_CAP_DEG
                                  # deg: the tilt the clamp actually enforces,
                                  # overriding what the aperture geometry would
                                  # allow. 0 = use the geometry. See
                                  # trocar.TILT_CAP_DEG for why it is not the
                                  # geometric limit
    bite_guard_mm: float = GUARD_OFF   # mm: when the MEASURED shaft bites
                                  # deeper than this into the rubber, act.
                                  # GUARD_OFF (-99) disables it; ordinary
                                  # negative values are legal and mean "keep
                                  # this many mm of CLEARANCE", which is what
                                  # a hard promise actually wants -- 0.0 only
                                  # promises not to have touched yet.
                                  # This acts on the COMMAND -- the same place
                                  # the cone clamp acts, which is stable -- and
                                  # NOT on the IK's RCM target, which is what
                                  # `rcm_fb` did and what rang (see the study's
                                  # 2.4(a)). It is the only term in this class
                                  # that makes the 6 mm hole a hard promise
                                  # rather than a well-tracked intention.
    bite_guard_tau: float = 0.15  # s: time constant of that tightening
    bite_guard_max_deg: float = 0.0    # deg of tilt the guard may take away.
                                  # DEFAULT 0 -- the tilt lever is MEASURED to
                                  # be unsafe and is kept only for reproducing
                                  # that. At 0.46 m it never engages and is
                                  # invisible; at 0.60 m, where the shaft does
                                  # touch, all three guard rows of the `rig`
                                  # screen ran away: latency 727 ms, RMS
                                  # 14.9 mm, bite p95 +12.9 mm on a 3 mm hole.
                                  # The loop is: bite > 0 -> tighten the cone
                                  # -> the command is rewritten away from where
                                  # the operator pointed -> tracking error
                                  # explodes -> lateral offset explodes -> bite
                                  # worse. It is the same positive feedback
                                  # `rcm_fb` had, one stage further out.
                                  # The reason is specific and worth keeping:
                                  # the tilt lever removes tilt the reference
                                  # LEGITIMATELY needs (the bench reaches 27
                                  # deg and the aperture allows 33.6), while
                                  # the thing actually eating the aperture at a
                                  # long stand-off is lateral offset, which
                                  # tilt does not touch. It pays a large cost
                                  # for a variable that is not the problem.
                                  # `bite_guard_slow` acts on the variable that
                                  # IS the problem and is swept separately.
    bite_guard_slow: float = 0.0  # 0..1: the guard's SECOND lever. A biting
                                  # shaft has two causes and the tilt cone only
                                  # fixes one. The other is lateral offset, and
                                  # offset is differential joint lag, which is
                                  # proportional to command SPEED -- so the
                                  # remedy is to take speed away. This is the
                                  # fraction of the command rate the guard may
                                  # withhold (0 = never, 1 = may stop the tip
                                  # dead). Slowing the operator's hand is a
                                  # real cost; leaning on the rubber is not
                                  # negotiable, so the trade is the right way
                                  # round.

    # ---- the virtual arm vs the arm that exists ----
    # The IK integrates its OWN joint state (`TipController.q`) and never looks
    # at what PhysX reached. Its shaft goes through the port by construction,
    # so the RCM residual it reports is its own, not the hardware's. Over a
    # short bench the two stay together; over a minute of teleop the redundant
    # (self-motion) directions drift apart without limit, because NOTHING in
    # the loop observes them -- the tip task cannot see the elbow. That drift
    # is what walks the real shaft off the hole and what makes the arm feel
    # less responsive the longer an episode runs.
    reseed_hz: float = 0.0        # corner frequency of the washout that pulls
                                  # the virtual arm onto the MEASURED one.
                                  # 0 = off (today). 0.2 Hz ~ 0.8 s time
                                  # constant. Large values approach a hard
                                  # re-seed every tick.
    reseed_mode: str = "null"     # "null": correct only the component that
                                  # lies in the null space of the tip+RCM task
                                  # -- i.e. exactly the self-motion that drifts
                                  # -- so the task servo loop is untouched and
                                  # this cannot cost tracking. "full": correct
                                  # everything, which does put PhysX inside the
                                  # position loop.
    reseed_every: int = 8         # ticks between refreshes of that projector
                                  # (the posture moves far slower than the tip)

    # ---- shaft roll: a task in the null space, or the joint itself ----
    roll_mode: str = "direct"     # "task": solve the roll as level 3 of the IK,
                                  # inside null(RCM+tip), through a damped
                                  # inverse (what shipped until 2026-07-30).
                                  # "direct": add the increment to joint 7,
                                  # which IS the roll axis.
                                  #
                                  # Measured on both arms' shipped start pose:
                                  # rotating joint 7 alone by 57.3 deg rolls the
                                  # shaft by 57.30 deg (ratio 1.0000), tilts the
                                  # shaft axis by 0.0000 deg -- so the RCM is
                                  # untouched to machine precision -- and moves
                                  # the tip 0.121 mm, which is just the tool's
                                  # 0.03 mm off-axis eccentricity turning. The
                                  # tip task absorbs that.
                                  #
                                  # "task" was slow for four stacked reasons:
                                  # `roll_clip` clips the roll ERROR to 5.7 deg
                                  # so it is a rate-limited slew rather than a
                                  # proportional follower; `roll_cap` caps the
                                  # joint step; `qd_max` then scales the WHOLE
                                  # dq, so the roll ceiling is qd_max = 2 rad/s
                                  # = 115 deg/s AND it is shared with the tip
                                  # task; and the 45 deg anti-windup discards
                                  # anything beyond that, which is what made a
                                  # fast twist go dead. A hand twists a pen at
                                  # 200-400 deg/s.
                                  #
                                  # "direct" also removes the reason the
                                  # RCM-health roll gate exists: the null-space
                                  # solve was free to spend the roll on j5/j6,
                                  # orbiting the ~9 cm wrist against tissue
                                  # (measured live: the physical RCM blew out to
                                  # 78 mm). Joint 7 cannot do that.
    roll_qd_max: float = 5.0      # rad/s: direct mode's own speed limit, kept
                                  # OFF the shared qd_max budget (qd_max is
                                  # 2.0 and is shared with the tip task, which
                                  # is what capped the roll at 115 deg/s).
                                  # 5.0 sits just under joint 7's OWN velocity
                                  # limit, which the URDF gives as 5.26 rad/s
                                  # = 301 deg/s -- NOT the 2.62 rad/s that
                                  # joints 1-4 have. An earlier value of 2.5
                                  # here was written believing joint 7 was
                                  # 2.6 rad/s like the shoulder; that threw
                                  # away more than half the joint's speed and
                                  # put the 1:1 ceiling at 143 deg/s when the
                                  # hardware allows 301. A hand twists a pen at
                                  # 200-400 deg/s, so those two numbers are on
                                  # opposite sides of the operator.

    # ---- posture: bounded objectives, not free integrators ----
    k_home: float = 0.0           # pull the redundancy back toward the START
                                  # posture. The manipulability climb and the
                                  # stop repulsion are pure ascent terms: they
                                  # integrate for as long as the episode runs
                                  # and nothing returns them. This bounds them.
    k_elbow: float = 0.0          # raise the joint-4 origin in world z, live
                                  # (start_pose.solve_start_q already does this
                                  # for the START pose only, so the elbow the
                                  # operator sees at t=0 is not the one they
                                  # have at t=60 s)
    elbow_point: int = 3          # which chain.fk_points() index that is

    # ---- joint servo (the PhysX articulation PD) ----
    kp: float = 6.0e4             # drive stiffness
    kd_ratio: float = 1.0 / 15.0  # kd = kd_ratio * kp  (seconds)
    vff: float = 0.0              # joint-velocity feed-forward, 0..1
    vff_lpf_hz: float = 0.0       # low-pass on the feed-forward, 0 = off
    null_lpf_hz: float = 0.0      # low-pass on the null-space levels, 0 = off
    qd_max: float = 2.0           # commanded joint-speed clamp, rad/s

    # ---- command shaping ----
    lead_s: float = 0.0           # feed the command forward by v_cmd * lead_s
    cmd_lpf_hz: float = 0.0       # 0 = off; first-order low-pass on the command
    adm_gain: float = 0.0         # admittance: metres of give per newton
    adm_tau: float = 0.05         # admittance relaxation time constant, s
    adm_k_probe: float = 80.0     # N/m: lag -> interaction-force proxy

    name: str = ""

    @property
    def kd(self) -> float:
        return self.kd_ratio * self.kp

    def label(self) -> str:
        if self.name:
            return self.name
        return (f"{self.rcm_mode}"
                f"{'' if self.rcm_mode == 'hard' else f'w{self.rcm_weight:g}'}"
                f"_d{self.damp:g}_i{self.iters}_kp{self.kp:.0e}"
                f"_kdr{self.kd_ratio:.3f}_v{self.vff:g}"
                f"{'' if self.null_lpf_hz <= 0 else f'n{self.null_lpf_hz:g}'}"
                f"{'' if self.vff_lpf_hz <= 0 else f'f{self.vff_lpf_hz:g}'}"
                f"_qd{self.qd_max:g}"
                f"_ld{self.lead_s*1000:.0f}"
                f"{'' if self.cmd_lpf_hz <= 0 else f'_lp{self.cmd_lpf_hz:g}'}"
                f"{'' if self.adm_gain <= 0 else f'_adm{self.adm_gain:g}'}"
                f"{'' if self.reseed_hz <= 0 else f'_rs{self.reseed_hz:g}{self.reseed_mode[0]}'}"
                f"{'' if self.k_home <= 0 else f'_hm{self.k_home:g}'}"
                f"{'' if self.k_elbow <= 0 else f'_eb{self.k_elbow:g}'}"
                f"{'' if self.bite_guard_mm <= GUARD_OFF else f'_bg{self.bite_guard_mm:g}'}"
                f"{'' if self.cone_mm < 0 else f'_cn{self.cone_mm:g}'}")

    def dict(self):
        return asdict(self)

    def replace(self, **kw):
        d = asdict(self)
        d.update(kw)
        d["name"] = kw.get("name", "")
        return ControlCfg(**d)


def cfg_fields():
    return [f.name for f in fields(ControlCfg) if f.name != "name"]


# --------------------------------------------------------------------------
# the IK step, both RCM formulations
# --------------------------------------------------------------------------
def ik_step_soft(chain, q, T_base, p_cmd, p_rcm, w_rcm, r_cmd=None,
                 iters=3, damp=2e-3, damp_roll=5e-3, roll_clip=0.1,
                 k_manip=0.05, k_limit=1.0, limit_margin=math.radians(15.0),
                 g_extra=None, post_cap=0.01, roll_cap=0.02, rcm_weight=1.0,
                 null_lpf_a=0.0, null_state=None, null_acc=None):
    """SOFT-constraint counterpart of laparo_ik.ik_step.

    The tip rows and the RCM rows go into ONE damped-least-squares task, the RCM
    rows scaled by `rcm_weight`. There is no priority: a tip command CAN trade
    RCM error away, in exchange for never fighting the constraint. This is the
    classic "penalty/soft RCM" of the literature and the thing the strict
    hierarchy replaced -- it is in the sweep so the choice is measured, not
    asserted.

    Roll and posture still run in the null space of the stacked task, so the
    only difference from the hard version is how the RCM is enforced.

    `null_lpf_a` (0..1) low-passes the NULL-SPACE contribution -- the roll and
    posture steps -- across calls, leaving the task levels untouched. A live
    session reported the arm visibly twisting; instrumenting it showed the
    joints moving 100-500x more high-frequency than the tip, i.e. the arm
    reconfiguring without the tip going anywhere, which is by definition
    null-space motion. Those two levels are comfort objectives with no
    bandwidth requirement (they only have to drift the elbow), so filtering
    them costs nothing the operator can feel -- unlike filtering the velocity
    feed-forward, which was tried first and made things worse because it broke
    the consistency between the position and velocity targets."""
    n = chain.n
    I = np.eye(n)
    gp = posture_gradient(chain, q, T_base, p_rcm, k_manip, k_limit,
                          limit_margin)
    if g_extra is not None:
        gp = gp + g_extra
    wb = float(np.clip(w_rcm, 0.0, 1.0)) * rcm_weight
    dq_null_call = np.zeros(n)          # roll + posture applied this call
    for it in range(iters):
        e0, phi0, J, Jr = arm_jacobians(chain, q, T_base, p_cmd, p_rcm,
                                        w_rcm, r_cmd)
        # arm_jacobians already scales the RCM rows by w_rcm; apply the extra
        # soft weight on top so rcm_weight alone sets the trade-off.
        Js = np.vstack([J[:3], wb * J[3:] / max(w_rcm, 1e-9)])
        es = np.concatenate([e0[:3], wb * e0[3:] / max(w_rcm, 1e-9)])
        A = Js @ Js.T + damp * np.eye(6)
        dq = Js.T @ np.linalg.solve(A, es)
        N = I - range_projector(Js)
        if r_cmd is not None:
            jr2 = N @ Jr
            s = float(Jr @ jr2) + damp_roll
            dq3 = jr2 * ((np.clip(phi0, -roll_clip, roll_clip)
                          - float(Jr @ dq)) / s)
            m3 = float(np.abs(dq3).max())
            if m3 > roll_cap:
                dq3 *= roll_cap / m3
            d3 = dq3 * limit_scale(q + dq, dq3, chain.lo, chain.hi)
            dq = dq + d3
            dq_null_call = dq_null_call + d3
            N = N @ (I - range_projector((Jr @ N)[None, :]))
        if it == 0:
            dq4 = N @ gp
            m4 = float(np.abs(dq4).max())
            if m4 > post_cap:
                dq4 *= post_cap / m4
            d4 = dq4 * limit_scale(q + dq, dq4, chain.lo, chain.hi)
            dq = dq + d4
            dq_null_call = dq_null_call + d4
        m = float(np.abs(dq).max())
        if m > 0.15:
            dq *= 0.15 / m
        q = np.clip(q + dq, chain.lo + 1e-3, chain.hi - 1e-3)
    # Replace this call's raw null-space motion with a low-passed version of it.
    # Done once per call (not per iteration) so the cutoff means what it says.
    if null_lpf_a > 0.0 and null_state is not None:
        null_state += (dq_null_call - null_state) * null_lpf_a
        q = np.clip(q - dq_null_call + null_state,
                    chain.lo + 1e-3, chain.hi - 1e-3)
    if null_acc is not None:
        null_acc += dq_null_call
    return q


# --------------------------------------------------------------------------
# the controller
# --------------------------------------------------------------------------
class TipController:
    """Tip command -> joint targets, for one arm.

    Call `step(p_cmd, v_cmd, p_meas, q_meas, dt, ...)` once per control tick.
    Returns (q_target, qd_target) ready for an ArticulationAction. The caller
    owns the articulation: this class never touches Isaac.
    """

    GUARD_V = 0.10        # m/s: the spec top speed the guard's cap scales down
                          # from, so `bite_guard_slow` reads as "the fraction
                          # of full speed I may take away"

    def __init__(self, cfg: ControlCfg, chain, T_base, p_rcm, q0,
                 min_depth=0.10, port_normal=None):
        from laparo_ik import ik_step
        self.cfg = cfg
        self.chain = chain
        self.T_base = T_base
        self.p_rcm = np.asarray(p_rcm, np.float64)
        self.q = np.asarray(q0, np.float64).copy()
        self.min_depth = min_depth
        self._ik_hard = ik_step
        self.p_shaped = None          # low-pass state
        self.x_adm = np.zeros(3)      # admittance give
        self.qd = np.zeros(len(self.q))
        self.qd_ff = np.zeros(len(self.q))   # filtered feed-forward
        self.null_state = np.zeros(len(self.q))  # null-space LPF state
        # --- aperture ---
        self.port_normal = (None if port_normal is None else
                            np.asarray(port_normal, np.float64)
                            / np.linalg.norm(port_normal))
        self.tilt_max = (0.0 if self.port_normal is None else
                         trocar.clamp_tilt_rad(cfg.cone_mm, cfg.tilt_cap_deg))
        self.rcm_err = np.zeros(3)    # measured shaft-vs-port error (LPF state)
        self.rcm_bias = np.zeros(3)   # what that feedback adds to the target
        self.n_cone = 0               # ticks the cone clamp bit
        self.n_clamp = 0              # ticks the joint-speed clamp bit
        # --- the 6 mm hole as a hard promise ---
        self.cone_extra = 0.0         # rad of tilt the bite guard has taken
        self.slow = 0.0               # 0..1 of command rate it has withheld
        self.p_slow = None            # rate-limiter state
        self.bite_meas = float("-inf")   # last MEASURED bite, m
        self.n_guard = 0              # ticks the bite guard was engaged
        # --- virtual arm vs measured arm ---
        self.q_home = np.asarray(q0, np.float64).copy()
        self.n_reseed = 0             # ticks since the projector was refreshed
        self._null_proj = None
        self.n_manip = 0              # ticks since the climb was recomputed
        self._manip_g = None          # ... and the value being held
        self.reseed_norm = 0.0        # |q_virtual - q_meas|, the drift itself
        # --- shaft roll ---
        self.roll_applied = 0.0       # rad of roll the last call actually put on
        self.roll_sat = 0             # ticks the roll hit a joint/speed limit

    # -- command shaping ---------------------------------------------------
    def shape(self, p_cmd, v_cmd, p_meas, dt):
        """Lead -> admittance -> low-pass, in that order.

        lead:       cancels the servo's velocity lag by asking for where the
                    operator WILL be, which is the only way a position-tracking
                    scheme reaches zero steady-state error at constant speed.
        admittance: lets the tip give way to an interaction force (estimated
                    from the tracking lag, the same proxy the haptics uses), so
                    contact does not turn into a rising drive force.
        low-pass:   trades latency for smoothness; in the sweep it is what a
                    vibration-first controller would reach for.
        """
        c = self.cfg
        p = np.asarray(p_cmd, np.float64).copy()
        if c.lead_s > 0.0 and v_cmd is not None:
            p = p + c.lead_s * np.asarray(v_cmd, np.float64)
        if c.adm_gain > 0.0:
            f = c.adm_k_probe * (np.asarray(p_meas, np.float64) - p)
            x_t = c.adm_gain * f
            a = dt / max(c.adm_tau, 1e-4)
            self.x_adm = self.x_adm + (x_t - self.x_adm) * min(a, 1.0)
            p = p + self.x_adm
        if c.cmd_lpf_hz > 0.0:
            if self.p_shaped is None:
                self.p_shaped = p.copy()
            a = 1.0 - math.exp(-2.0 * math.pi * c.cmd_lpf_hz * dt)
            self.p_shaped = self.p_shaped + (p - self.p_shaped) * a
            p = self.p_shaped.copy()
        else:
            self.p_shaped = p.copy()
        # minimum insertion: keep the direction, floor the depth (the RCM makes
        # tip-to-port distance the insertion depth)
        if self.min_depth > 0.0:
            v = p - self.p_rcm
            d = float(np.linalg.norm(v))
            if d < self.min_depth and d > 1e-6:
                p = self.p_rcm + v / d * self.min_depth
        # aperture cone: the same idea as the depth floor, on the other DOF.
        # Under the RCM the shaft direction IS the port-to-tip direction, so a
        # tip command more than `tilt_max` off the port normal is a command for
        # the shaft to lie ON the rubber -- the arm cannot honour it and simply
        # leans on the sheet. Clamp it where the command is made instead, and
        # reserve `cone_mm` of the aperture for the tracking error that is
        # still to come (see trocar.max_tilt_rad).
        # The bite guard rides on the SAME clamp: it is the cone, tightened by
        # however much the measured shaft is currently over-running the hole.
        # Reserving room in advance (cone_mm) handles the error you can predict;
        # this handles the error you cannot, and it is the term that makes the
        # aperture a constraint rather than a well-tracked intention.
        tmax = self.tilt_max
        if c.bite_guard_mm > GUARD_OFF and self.port_normal is not None:
            base = (self.tilt_max if self.tilt_max > 0.0
                    else trocar.max_tilt_rad(0.0))
            tmax = max(math.radians(4.0), base - self.cone_extra)
        if tmax > 0.0:
            p, hit = trocar.cone_clamp(p, self.p_rcm, self.port_normal, tmax)
            self.n_cone += int(hit)
        # Second lever: a SPEED CAP while the shaft is biting. Last in the
        # chain, so what it limits is the command the arm will actually be
        # asked for -- cone clamp and depth floor included.
        #
        # A cap, not a low-pass: a filter only delays the command and the tip
        # still arrives at the same offset a moment later, which is no use when
        # the offset IS the violation. A cap removes the speed, and lateral
        # offset at the port is differential joint lag, which is proportional
        # to it. It also gives the position back: the moment the operator slows
        # below the cap the command catches up exactly, so nothing accumulates.
        if self.slow > 0.0 and self.p_slow is not None:
            v_cap = self.GUARD_V * max(1.0 - self.slow, 0.02)
            d = p - self.p_slow
            n = float(np.linalg.norm(d))
            if n > v_cap * dt:
                p = self.p_slow + d * (v_cap * dt / n)
        self.p_slow = p.copy()
        return p

    # -- the aperture guard ------------------------------------------------
    def update_bite_guard(self, line_meas, dt):
        """Watch the MEASURED shaft against the real hole and decide how much
        of the tilt cone and of the command speed to take away. Called once per
        tick, before `shape()`, which applies both.

        `line_meas` is (tip, axis) from the joint angles PhysX actually reached,
        so `bite` here is the hardware's number, not the solver's residual.

        Two levers because a biting shaft has two causes and each lever only
        answers one: tilt eats the aperture geometrically, lateral offset eats
        it through differential joint lag. Measured off-engine, the tilt lever
        alone moved a 2.55 mm bite to 2.51; the speed cap took it to 0.88.

        Both engage and release on the same first-order lag (`bite_guard_tau`),
        so the guard cannot chatter at the boundary the way a hard switch
        would, and it hands the speed back as soon as the shaft is clear.
        """
        c = self.cfg
        if c.bite_guard_mm <= GUARD_OFF or line_meas is None \
                or self.port_normal is None:
            return
        pt, am = np.asarray(line_meas[0]), np.asarray(line_meas[1])
        g = trocar.port_geometry(self.p_rcm, self.port_normal, pt[None, :],
                                 am[None, :])
        self.bite_meas = float(g["bite"][0])
        over = self.bite_meas - c.bite_guard_mm * 1e-3
        a = min(1.0, max(dt, 1e-6) / max(c.bite_guard_tau, 1e-4))
        lim = math.radians(c.bite_guard_max_deg)
        if over > 0.0:
            # how much tilt would have to go to buy back `over` of margin:
            # d(margin)/d(tilt) = -r sin(t)/cos^2(t); use the local slope, and
            # let the lag do the rest rather than trusting a one-shot inversion
            t = max(math.radians(5.0), min(math.acos(float(np.clip(
                np.dot(am, self.port_normal), -1.0, 1.0))),
                math.radians(60.0)))
            slope = trocar.SHAFT_R * math.sin(t) / max(math.cos(t) ** 2, 1e-3)
            want = self.cone_extra + over / max(slope, 1e-3)
            self.cone_extra += (min(want, lim) - self.cone_extra) * a
            self.slow += (c.bite_guard_slow - self.slow) * a
            self.n_guard += 1
        else:
            self.cone_extra += (0.0 - self.cone_extra) * a
            self.slow += (0.0 - self.slow) * a
        self.cone_extra = float(np.clip(self.cone_extra, 0.0, lim))
        self.slow = float(np.clip(self.slow, 0.0, 0.999))

    # -- keep the virtual arm on the real one -------------------------------
    def reseed(self, q_meas, dt, p_cmd=None, w_rcm=1.0):
        """Wash the IK's own joint state toward the MEASURED one.

        Called once per tick, before `step()`. Without it `self.q` is a free
        integrator: it is seeded once at start-up and from then on only ever
        adds its own increments, so the pose the RCM is enforced on is the pose
        the controller BELIEVES it has. The believed arm and the real arm agree
        to first order in the task directions -- the servo closes that loop --
        but in the REDUNDANT directions nothing does, and over a minute the two
        end up in visibly different self-motions: the believed shaft threads the
        hole, the real one leans on the rubber, and the real elbow is wherever
        the accumulated null-space steps left it.

        `reseed_mode="null"` corrects ONLY those unobserved directions. The
        component along the tip+RCM task is left to the servo that already owns
        it, so this cannot add tracking lag -- which is what makes it a
        different animal from `rcm_fb`, the measured-error term that rang.
        """
        c = self.cfg
        if q_meas is None:
            return
        d = np.asarray(q_meas, np.float64) - self.q
        self.reseed_norm = float(np.linalg.norm(d))
        if c.reseed_hz <= 0.0:
            return
        if c.reseed_mode == "null":
            if (self._null_proj is None
                    or self.n_reseed % max(c.reseed_every, 1) == 0):
                J = arm_jacobians(self.chain, self.q, self.T_base,
                                  self.chain.fk(self.q, self.T_base)[0]
                                  if p_cmd is None else np.asarray(p_cmd),
                                  self.p_rcm, w_rcm)[2]
                # rank 5 BY CONSTRUCTION (tip 3 + RCM lateral 2). Taking it
                # analytically rather than by threshold keeps the projector
                # from flickering on the RCM's ~1e-5 finite-difference third
                # singular value -- see range_projector's own note.
                self._null_proj = (np.eye(self.chain.n)
                                   - range_projector(J, rank=5))
            self.n_reseed += 1
            d = self._null_proj @ d
        a = min(1.0, 1.0 - math.exp(-2.0 * math.pi * c.reseed_hz
                                    * max(dt, 1e-9)))
        self.q = self.q + a * d

    def freeze(self, p_hold, q_meas=None):
        """Stop the tip task dead at `p_hold`, without touching the posture.

        Called when the operator lets go of the clutch. Everything that can
        still carry motion forward is re-anchored on the point being held:

          - the shaping states (`p_slow`, `p_shaped`), or the rate limiter
            would spend the next ticks travelling from the abandoned command;
          - the velocity feed-forward, which is a JOINT-SPEED command and
            would otherwise keep being applied for as long as it takes to
            decay;
          - the virtual arm itself, if `q_meas` is given. This is the one that
            matters. `self.q` leads the real arm by the tracking lag -- that
            lead IS how a position servo produces motion -- and it is handed
            to the drive as the target, so on release the real arm keeps
            flying until it has closed the whole lead. Seeding it with the
            measured pose makes the target the pose the arm is already in.

        The null-space objectives are deliberately left running: they act on
        `self.q` after this, so the elbow keeps optimising while the tip
        stands still, which is what a released clutch should look like."""
        p = np.asarray(p_hold, np.float64).copy()
        self.p_slow = p.copy()
        self.p_shaped = p.copy()
        self.x_adm = np.zeros(3)
        self.qd = np.zeros(len(self.q))
        self.qd_ff = np.zeros(len(self.q))
        self.rcm_err = np.zeros(3)
        if q_meas is not None:
            self.q = np.asarray(q_meas, np.float64).copy()

    # -- shaft roll, driven on the joint that IS the roll axis ---------------
    ROLL_JOINT = 6                # fr3_joint7: its axis is the flange normal,
                                  # which is the tool shaft axis (verified: the
                                  # shaft axis tilts 0.0000 deg under it)

    def direct_roll(self, droll, dt):
        """Put `droll` radians of shaft roll straight onto joint 7.

        Returns how much was ACTUALLY applied, which is the requested amount
        unless the joint's own speed limit or its travel limit got in the way.
        The caller must advance its roll target by the returned value, not by
        the requested one -- then a saturated roll cannot wind up, by
        construction, and the 45 deg anti-windup the task mode needs becomes
        unnecessary.

        Called BEFORE the IK solve so that `q_prev` already contains it: that
        keeps the roll off the shared `qd_max` budget (which would otherwise
        throttle it to 2 rad/s and make it compete with the tip task) while
        still letting the tip/RCM levels correct the 0.002 mm/deg eccentricity
        the tool's off-axis tip produces.
        """
        c = self.cfg
        if c.roll_mode != "direct" or not droll:
            return 0.0
        j = self.ROLL_JOINT
        step = float(np.clip(droll, -c.roll_qd_max * max(dt, 1e-9),
                             c.roll_qd_max * max(dt, 1e-9)))
        buf = 1e-3                       # same stop buffer the IK levels use
        lo, hi = self.chain.lo[j] + buf, self.chain.hi[j] - buf
        applied = float(np.clip(self.q[j] + step, lo, hi) - self.q[j])
        self.q[j] += applied
        self.roll_applied = applied
        self.roll_sat += int(abs(applied) < abs(step) - 1e-12)
        return applied

    # -- posture objectives this class owns ---------------------------------
    def posture_extra(self):
        """`k_home` + `k_elbow`, as a joint-space direction for ik_step's
        posture level. Both are BOUNDED: the home pull vanishes at the home
        pose and the elbow term saturates once the elbow is as high as the
        tip task allows. The two terms already in `posture_gradient`
        (manipulability climb, stop repulsion) are pure ascent and integrate
        for as long as the episode lasts."""
        c = self.cfg
        g = None
        if c.k_home > 0.0:
            g = c.k_home * (self.q_home - self.q)
        if c.k_elbow > 0.0:
            ge = elbow_up_gradient(self.chain, self.q, self.T_base,
                                   c.k_elbow, c.elbow_point)
            g = ge if g is None else g + ge
        return g

    def step(self, p_cmd, v_cmd, p_meas, dt, w_rcm=1.0, r_cmd=None,
             g_extra=None, line_meas=None, q_meas=None, droll=None):
        """`line_meas` = (tip, axis) of the MEASURED shaft, i.e. from the joint
        angles PhysX actually reached. Without it the RCM is open loop: the IK
        integrates its own virtual arm, whose shaft goes through the port by
        construction, and nothing ever tells the controller that the real shaft
        does not. `rcm_fb` closes that loop at the TARGET (and rang);
        `bite_guard_mm` closes it at the COMMAND; `q_meas` + `reseed_hz` close
        it at the STATE, which is where the divergence actually lives.

        Pass `q_meas` (the measured joint angles) every tick. It is optional
        only so that off-engine replays without a plant still run; with
        `reseed_hz > 0` and no `q_meas` the drift correction silently does
        nothing, and `reseed_norm` -- the drift itself -- is not observable."""
        c = self.cfg
        self.reseed(q_meas, dt, p_cmd=p_cmd, w_rcm=w_rcm)
        # Roll FIRST, and onto the joint: before `q_prev` is taken, so the roll
        # is not counted in `dq` and therefore not throttled by `qd_max`.
        self.roll_applied = 0.0
        if droll:
            self.direct_roll(droll, dt)
        self.update_bite_guard(line_meas, dt)
        p_eff = self.shape(p_cmd, v_cmd, p_meas, dt)
        gx = self.posture_extra()
        if gx is not None:
            g_extra = gx if g_extra is None else g_extra + gx

        # --- outer loop on the measured shaft-vs-port error ---
        # The tip has a servo behind it (position + velocity feed-forward); the
        # port constraint had nothing. What lands the shaft off the hole is not
        # bulk delay -- a pure time lag slides the shaft ALONG itself and keeps
        # the crossing point -- it is DIFFERENTIAL lag between joints, which no
        # amount of feed-forward on the tip removes. So measure it and bias the
        # RCM target the other way.
        p_rcm_eff = self.p_rcm
        if c.rcm_fb > 0.0 and line_meas is not None:
            pt, am = np.asarray(line_meas[0]), np.asarray(line_meas[1])
            p_c = pt + am * float(np.dot(self.p_rcm - pt, am))
            err = p_c - self.p_rcm            # where the real shaft crosses
            if c.rcm_fb_lpf_hz > 0.0:
                a = 1.0 - math.exp(-2.0 * math.pi * c.rcm_fb_lpf_hz
                                   * max(dt, 1e-9))
                self.rcm_err = self.rcm_err + (err - self.rcm_err) * a
            else:
                self.rcm_err = err
            bias = -c.rcm_fb * self.rcm_err * float(np.clip(w_rcm, 0.0, 1.0))
            nb = float(np.linalg.norm(bias))
            if nb > c.rcm_fb_max:
                bias = bias * (c.rcm_fb_max / nb)
            self.rcm_bias = bias
            p_rcm_eff = self.p_rcm + bias

        # THE MANIPULABILITY CLIMB, HELD BETWEEN REFRESHES.
        # `ik_step_soft` adds `g_extra` to the posture gradient by plain sum,
        # so handing the climb over that way and zeroing k_manip inside is
        # arithmetically the same term -- just evaluated at a q that is up to
        # one refresh interval old, which for an objective that steers the
        # elbow is nothing. What this buys is the whole 64-FK block on most
        # ticks, and it is the biggest single item in the control tick -- the
        # one that DOUBLES when both arms are being flown.
        k_manip_ik = c.k_manip
        if c.manip_hz > 0.0 and c.k_manip > 0.0:
            every = max(1, int(round(1.0 / (c.manip_hz * max(dt, 1e-9)))))
            if self._manip_g is None or self.n_manip % every == 0:
                self._manip_g = manip_gradient(self.chain, self.q, self.T_base,
                                               self.p_rcm, c.k_manip)
            self.n_manip += 1
            g_extra = (self._manip_g if g_extra is None
                       else g_extra + self._manip_g)
            k_manip_ik = 0.0
        kw = dict(iters=c.iters, damp=c.damp, k_manip=k_manip_ik,
                  k_limit=c.k_limit,
                  limit_margin=math.radians(c.limit_margin_deg),
                  g_extra=g_extra, post_cap=c.post_cap, roll_cap=c.roll_cap)
        q_prev = self.q.copy()

        def solve(p_target, q_from, null_acc=None):
            if c.rcm_mode == "soft":
                a_null = (1.0 - math.exp(-2.0 * math.pi * c.null_lpf_hz
                                         * max(dt, 1e-9))) \
                    if c.null_lpf_hz > 0 else 0.0
                return ik_step_soft(self.chain, q_from, self.T_base, p_target,
                                    p_rcm_eff, w_rcm, r_cmd,
                                    rcm_weight=c.rcm_weight,
                                    null_lpf_a=a_null,
                                    null_state=self.null_state,
                                    null_acc=null_acc, **kw)
            return self._ik_hard(self.chain, q_from, self.T_base, p_target,
                                 p_rcm_eff, w_rcm, r_cmd, null_acc=null_acc,
                                 **kw)

        null_acc = np.zeros(len(self.q))
        q = solve(p_eff, self.q, null_acc)
        # commanded joint-speed clamp: uniform scale, not a per-joint clip --
        # box-clipping bends the motion direction and re-couples the IK levels
        dq = q - q_prev
        m = float(np.abs(dq).max())
        lim = c.qd_max * max(dt, 1e-6)
        if m > lim:
            self.n_clamp += 1
            if c.clamp_mode == "null":
                # Spend the budget by PRIORITY. The roll and posture levels are
                # comfort objectives living in the null space of the tip+RCM
                # task, so scaling them back does not disturb the constraint to
                # first order -- while scaling everything uniformly (the
                # "joint" mode) shrinks the RCM correction by the same factor,
                # which is how a clamp meant to protect the arm ends up walking
                # the shaft out of the port. Their caps (10 and 20 mrad) are
                # both bigger than the whole per-tick budget, so on this arm
                # they are usually the reason the clamp fired at all.
                dq_task = dq - null_acc
                lo, hi = 0.0, 1.0
                if float(np.abs(dq_task).max()) <= lim:
                    for _ in range(16):       # largest null fraction that fits
                        mid = 0.5 * (lo + hi)
                        if float(np.abs(dq_task + mid * null_acc).max()) <= lim:
                            lo = mid
                        else:
                            hi = mid
                    dq = dq_task + lo * null_acc
                else:                          # even the task alone overruns
                    dq = dq_task
                m = float(np.abs(dq).max())
                q = q_prev + dq
            elif c.clamp_mode == "task":
                # Scaling dq scales the RCM correction along with the tip step,
                # so the clamp ITSELF walks the shaft out of the port -- and it
                # bites exactly when the operator moves fast, which is when the
                # report says the shaft leaves the hole. Give up tip travel
                # instead: ask for a nearer tip and let the RCM rows keep their
                # full step.
                p_tip_i, _ = self.chain.fk(q_prev, self.T_base)
                q = solve(p_tip_i + (p_eff - p_tip_i) * (lim / m), q_prev)
                dq = q - q_prev
                m = float(np.abs(dq).max())
            if m > lim:                    # safety net, and the "joint" mode
                dq = dq * (lim / m)
                q = q_prev + dq
        self.q = q
        self.qd = dq / max(dt, 1e-6)
        if c.vff <= 0.0:
            return q, None, p_eff
        # LOW-PASS THE FEED-FORWARD. qd = dq/dt divides the IK's step by the
        # timestep, which at 480 Hz multiplies whatever numerical noise the
        # finite-difference Jacobian and the DLS solve leave in dq by 480 --
        # and PhysX then multiplies THAT by kd (3.3e4 here), so a 1 mrad/s
        # wobble becomes ~33 N.m of torque chatter. The feed-forward only has
        # to cancel the kd lag at the bandwidth the operator actually moves at
        # (a few Hz), so everything above that is pure noise injection.
        if c.vff_lpf_hz > 0.0:
            a = 1.0 - math.exp(-2.0 * math.pi * c.vff_lpf_hz * max(dt, 1e-9))
            self.qd_ff = self.qd_ff + (self.qd - self.qd_ff) * a
        else:
            self.qd_ff = self.qd
        return q, c.vff * self.qd_ff, p_eff


# --------------------------------------------------------------------------
# the swept result
# --------------------------------------------------------------------------
# The shipped controller. Three studies, in order (full report:
# CONTROLLER_STUDY.md):
#
#   Part 1  10 cm/s, free space, smooth command      -> soft RCM + vff + kp
#   Part 2  + the port is a 6 mm hole in stiff rubber -> command low-pass,
#           not a "pure loss" once the objective has a physical constraint in it
#   Part 3  + the whole speed range (2..100 mm/s), holding the organ, and the
#           command quantised the way the Touch's encoder quantises a hand
#
# Against the Part 1 controller, on the Part 3 bench (organ grasped, input
# quantised at one 0.055 mm encoder count):
#
#   shaft pressing on the rubber   31.0 % of ticks -> 0.00 %   (aperture bench)
#   bite worst                     +1.53 mm        -> -0.09 mm (never touches)
#   lurch at 2 mm/s                1.56x           -> 1.33x
#   vibration                      31.7 um         -> 8.0 um
#   RMS tracking                   0.46 mm         -> 0.17 mm
#   worst |latency| over 2..100 mm/s  4.2 ms       -> 2.6 ms
#
# Why each value:
#
#   rcm_mode="soft"  unchanged from Part 1: soft beats the strict hierarchy on
#                    tracking AND on the RCM itself. weight 1.0, not 10 -- 10
#                    won a single-factor screen and then lost the factorial,
#                    where it costs 1.8x the vibration.
#   vff=1.0          still the one that matters, and NOT negotiable: turning it
#                    off costs 29-35 ms at EVERY speed and 3.4 mm of error at
#                    10 cm/s. Part 1's finding survives all three studies.
#   vff_lpf_hz=30    what Part 1 rejected -- measured at 10 cm/s, where there is
#                    nothing to filter. vff is dq/dt, so with a quantised
#                    command it multiplies one encoder count by the tick rate:
#                    at 2 mm/s a 0.055 mm step is 13 ticks of real motion and
#                    reads as 13x the hand's speed. Filtering it costs ~0 ms and
#                    removes a third of the vibration.
#   cmd_lpf_hz=60    likewise rejected by Part 1 as "pure loss", and likewise
#                    only against an objective with no constraint in it. It is
#                    what keeps the shaft inside the 6 mm hole: unfiltered, the
#                    tool is on the rubber 31 % of the time. 60 rather than 30
#                    because with vff_lpf carrying the noise, 60 keeps the port
#                    (0.00 % touch, 0.09 mm of margin) at half the latency.
#   post_cap=0.002   the posture null-space step, 10 -> 2 mrad. Its old cap was
#                    bigger than the entire per-tick joint budget (qd_max/480 =
#                    4.2 mrad), so it was what tripped the speed clamp -- and
#                    the clamp scales the RCM correction away with everything
#                    else. 1 mrad is a step too far (the posture degrades).
#   kd_ratio=0.0333  free to choose once vff removes the latency term; this is
#                    the quieter of the two at rest, as Part 1 already noted.
#   kp=5e5           NOT the top row: kp 2e6 is 0.3 ms and 0.03 mm better. Same
#                    tie-break as Part 1, and it applies harder now that the
#                    bench actually holds tissue -- 12-30 kPa organs, and a real
#                    FR3's joints are nowhere near 2e6 stiff.
#
# Pruned to their defaults, each against a measurement: lead_s, adm_gain
# (destabilises at rest), qd_max (2.0 is a narrow optimum), rcm_fb (an outer
# loop on the measured shaft; halves the error against an ideal plant, diverges
# against PhysX at every gain from 0.05 to 1.5), clamp_mode (task/null both
# measured neutral for the aperture; "null" does halve the start transient and
# is there if wanted).
#
#   cone_mm=0.15     the aperture tilt clamp, ON since a live session reported
#                    the shaft walking out of the port the longer the clutch
#                    was held. It is geometry, not drift: hand travel x --scale
#                    is tip travel, tip travel at a fixed depth is TILT, and
#                    tilt is what closes the hole's lateral budget (0.43 mm at
#                    13 deg, 0.09 mm at 31 deg, zero at 33.6). Nothing stopped
#                    the operator asking for a tilt the 6 mm hole cannot pass.
#                    Reserving 0.15 mm caps the command at 28.7 deg, which is
#                    FREE on the task -- swept at 34/30/29 deg it never fires
#                    and the metrics are identical to no cone -- while 25 deg
#                    fires on 8 % of ticks and 22 deg is actively harmful
#                    (rms 0.16 -> 2.99 mm: clamping that hard makes the arm
#                    fight the command). Measured on the reported failure: a
#                    single long clutch stroke put the shaft on the rubber for
#                    30.3 % of ticks; with the cone, 0.00 %.
# Kept by value, not as an alias of BEST: it is the control group every later
# stage is measured against, so it has to survive BEST changing.
PART_B = ControlCfg(name="partB-winner-2026-07-29",
                    rcm_mode="soft", rcm_weight=1.0,
                    vff=1.0, kd_ratio=1.0 / 15.0, kp=5.0e5,
                    qd_max=2.0, lead_s=0.0, cmd_lpf_hz=0.0, adm_gain=0.0)

PART_3 = ControlCfg(name="sweep-winner-2026-07-29-p3",
                    rcm_mode="soft", rcm_weight=1.0,
                    vff=1.0, vff_lpf_hz=30.0, cmd_lpf_hz=60.0,
                    post_cap=0.002, kd_ratio=0.0333, kp=5.0e5,
                    qd_max=2.0, lead_s=0.0, adm_gain=0.0,
                    cone_mm=0.15)

# Part 4 (2026-07-30, PORT_OFFSET_STUDY.md). The rig reported two things:
# J4 sitting near its stop with the arm shaking there, and an episode that
# degrades as it runs. Two axes changed; everything Part 3 chose survived.
#
# Against Part 3, at the newly recommended 0.55 m stand-off, on the loaded
# bench (organ grasped and lifted, input quantised at one encoder count):
#
#   tip error worst              25.94 mm  ->  0.44 mm
#   RMS tracking                  4.37 mm  ->  0.21 mm
#   bite worst (+ = on the rubber) +38.38 mm -> +0.09 mm
#   worst joint-limit margin        13.7 deg -> 32.3 deg   (J2 -> J4)
#   vibration                      29.8 um  ->  10.2 um
#   lowest elbow                    387 mm  ->   639 mm
#
#   k_elbow=0.1   THE change. At 0.55 m every low-elbow posture loses the port
#                 during the loaded lift -- the tip goes 26-45 mm off command
#                 and the shaft is 19-38 mm outside a 3 mm hole -- while the
#                 speed rungs on either side of that segment track to 0.2 mm.
#                 This is the only posture that survives it. It also raises the
#                 elbow 387 -> 639 mm (the front view asked for that on its own)
#                 and the WORST joint margin 13.7 -> 32.3 deg.
#                 Note what it does NOT do: J4 itself goes slightly the other
#                 way (34.0 -> 32.3 deg at 0.55 m, 20.3 -> 15.2 at 0.46 m) --
#                 raising the elbow spends J4 to do it. The worst margin still
#                 improves because the binding joint was J2. **The J4 complaint
#                 is answered by the STAND-OFF, not by this term**: J4 has
#                 8.7 deg in the corners of the view at 0.46 m and 32-34 deg at
#                 0.55 m, and no posture setting moves the 0.46 m figure
#                 (all eight tried land within 0.1 deg of each other).
#                 NOT k_home: same elbow height, but pulling the WHOLE start
#                 posture back fights the tip task -- 2.3 % of ticks on the
#                 rubber at 0.46 m and 97.7 % at 0.60 m.
#                 NOT limit_margin_deg: it wins alone (13.7 -> 29.3 deg) and
#                 then adds NOTHING once k_elbow is on -- identical to three
#                 significant figures in the 2^3 factorial. Both terms push the
#                 same redundancy the same way and k_elbow pushes harder.
#   reseed_hz=1.0 Adopted for TRACKING UNDER LOAD, not for the drift it was
#   reseed_mode=  built for: 0.25 -> 0.21 mm RMS and 1.8 -> 1.3 % contact at
#     "full"      0.55 m, 37.6 -> 18.5 % at 0.60 m. The long-episode drift it
#                 was meant to fix DID NOT REPRODUCE (86 s of identical laps:
#                 |q_virtual - q_real| ends at 0.0002 rad and nothing degrades)
#                 -- so it is not claimed as that fix. `full` rather than
#                 `null` because the error being corrected is drive deflection,
#                 which lives in the TASK directions that `null` excludes.
#
# kp stays 5.0e5. At 0.55 m, kp=1e6 takes the contact from 1.3 % to 0.00 %,
# but whether the real FR3's joint impedance reaches 1e6 is not established,
# and 5e5 already leaves the worst bite at +0.09 mm against Part 3's +38.38.
# THE GAIN IS A DIAL, AND IT IS THE ONE JUDGEMENT CALL IN THIS STUDY.
# Two of the rig's requirements pull opposite ways on it, and the sweep measured
# both ends at 0.55 m (loaded = grasp+lift, free = 5-site tour):
#
#            loaded bite/touch    free bite/touch    elbow min (loaded/free)
#   0.05     +36.7 / 4.6 %        -0.05 / 0.0 %      418 / 384   <- loses the lift
#   0.10     -0.08 / 0.0 %        -0.06 / 0.0 %      467 / 389
#   0.30     +0.09 / 1.3 %        +1.46 / 2.9 %      639 / 628
#
# 0.10 is the default because the port was stated as non-negotiable and it is
# the only setting that gives 0.00 % contact on BOTH benches at the shipped
# stiffness -- and it still fixes the loaded lift (which 0.05 does not: bite
# +36.7 mm) and still raises the worst joint margin 13.7 -> 29.5 deg.
# The cost is honest: the elbow reaches 467 mm loaded / 389 mm free instead of
# 639 / 628. If the front view matters more than the last 1-3 % of port
# contact, use 0.3; if the drive can also take kp=1e6 then 0.3 gives BOTH on
# the loaded bench (bite -0.01, touch 0.00 %, elbow 638) and still 1.7 % on the
# free tour. `--k-elbow` sets it live.
BEST = PART_3.replace(name="rig-answer-2026-07-30-p4",
                      k_elbow=0.1, reseed_hz=1.0, reseed_mode="full")
