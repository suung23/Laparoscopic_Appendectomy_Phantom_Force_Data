#!/usr/bin/env python3
"""Bimanual laparoscopic surgical cell as a VLA evaluation environment.

This is the programmatic twin of `isaac/teleop_left_touch.py`. The teleop file
flies the two instruments from two Touch X styluses; this one flies them from a
POLICY, through the identical kinematics (`laparo_ik`), the identical controller
(`laparo_control.BEST`, the grid-search winner) and the identical scene
(`out/surgical_cell.usd`). Anything a policy is scored on here is measured on
the same rig the human operator drives.

    from vla_env import LaparoVLAEnv, Action
    env = LaparoVLAEnv(headless=True, control_hz=30, chunk=8)
    obs = env.reset()
    while not done:
        act = policy(obs)                       # (2, K, 7) float32
        obs, info = env.step(act)
    env.close()

--------------------------------------------------------------------------
ACTION  -- one chunk per step(), both arms
--------------------------------------------------------------------------
`(2, K, 7) float32`, arm 0 = LEFT (grasper), arm 1 = RIGHT (cutter), K =
`chunk` sub-steps of 1/`control_hz` seconds each:

    [:, :, 0:3]  dx, dy, dz  metres, in the TOOL-TIP frame of that arm
    [:, :, 3:6]  rx, ry, rz  radians, rotation vector in the same frame
    [:, :, 6]    gripper opening, 0 = fully closed .. 100 = fully open

The translation is exactly what the Touch stylus delivers today: a delta in the
instrument's own frame, applied to the tip POSITION command. The delta is
expressed in the tool frame captured at the start of each sub-step and
integrated, so a chunk is a path, not a jump.

The frame is the URDF `tool_tip` link, NOT a convention invented here, so it
matches the teleop demonstrations axis for axis:

    +x  the jaw CLOSING direction
    +y  the clevis PIN axis
    +z  BACK along the shaft, out of the patient  ->  INSERTION IS -z

The -z-forward convention is shared with the OpenHaptics stylus (push the pen
along its -z to insert). Watch the sign on dz: `dz = -0.001` drives the
instrument 1 mm FURTHER IN.

Of the three rotations only rz -- the ROLL about the shaft -- is physically
available: the trocar RCM pins the shaft line through a 6 mm hole, so the two
pitch/yaw DOF of the tip frame are determined by the tip position and cannot be
commanded independently. rx and ry are accepted, ignored, and reported back in
`info["<arm>"]["rot_rejected"]` so a policy that leans on them is not silently
mis-scored.

The gripper is CONTINUOUS: 0..100 maps linearly onto 0..`open_deg` of included
jaw angle (74 deg left grasper, 50 deg right cutter), both jaws driven
symmetrically and simultaneously. It slews at `gripper_rate_dps`; the achieved
angle is always reported next to the commanded one.

--------------------------------------------------------------------------
OBSERVATION
--------------------------------------------------------------------------
See `LaparoVLAEnv.observe()` and VLA_HANDOFF.md. Summary:

  obs["endoscope"]["rgb"]    (720, 1280, 3) uint8   -- the 30 deg scope
  obs["d405"]["rgb"]         (480,  848, 3) uint8   -- the overhead RealSense
  obs["d405"]["depth"]       (480,  848)    float32 -- metres, z to image plane
  obs["d405"]["rgb_right"]   (480,  848, 3) uint8   -- stereo partner (optional)
  obs["left"] / obs["right"]                        -- per-arm proprioception
  obs["left_force"]                                 -- the instrumented gripper

Every camera also carries `K` (3x3 pinhole intrinsics as the RTX renderer
actually projects), `T_world_cam` and `T_rig_cam` (4x4, OpenCV convention:
+x right, +y down, +z forward).

--------------------------------------------------------------------------
FRAMES
--------------------------------------------------------------------------
world   the USD stage frame; +z up, the table top at z = 0.
rig     THE FRAME POLICIES SHOULD USE. Origin at the midpoint of the two FR3
        base origins, +x from the left base toward the right base, +z world up,
        +y = z x x. `obs["T_world_rig"]` converts. It is fixed for a given
        cell, so it is a rigid relabelling of world -- but it is centred on the
        robot pair rather than on an arbitrary stage origin.
tool    per arm: the URDF `tool_tip` link frame -- +x jaw closing, +y clevis
        pin, +z back along the shaft out of the patient. Rolls with the
        instrument. Identical on both arms. See ArmRig.tool_axes().
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------
# defaults -- every one of these is a constructor keyword
# --------------------------------------------------------------------------
DEFAULTS = dict(
    scene="out/surgical_cell.usd",
    physics_hz=480.0,          # the rate the controller was tuned and swept at
    control_hz=30.0,           # one action sub-step; also the observation rate
    chunk=8,                   # action sub-steps per step() call
    insert=0.10,               # m past the port at reset. Tracks
                               # build_surgical_cell's --start-insert: this env
                               # RE-SOLVES the start pose rather than trusting
                               # the stage, so a mismatch silently spawns the
                               # tools somewhere else. This is the PARKED depth;
                               # the working pose is reached by the approach
                               # below, and it is from there that step() runs
    start_aim="approach",      # which direction each tool spawns pointing.
                               # "approach" = along the ray from its port to its
                               # working tip, so the spawn is that pose withdrawn
                               # and the approach is a pure slide in.
                               # "endoscope" = the old aim at the middle of the
                               # scope image, which is only sound while the tools
                               # are deeper than the camera
    start_max_tilt=60.0,       # deg: spawn-aim cone half-angle ("endoscope"
                               # only). Matches the operating cap
                               # (trocar.TILT_CAP_DEG)
    approach=True,             # fly both tips from the spawn to the measured
                               # working pose (start_pose.APPROACH_TIP_WORLD)
                               # during reset(), before the first observation is
                               # returned. The policy therefore never sees, and
                               # never has to spend actions on, the insertion --
                               # every episode begins with the instruments in
                               # frame over the appendix
    approach_speed=0.03,       # m/s along the approach arc
    approach_tol_m=0.002,      # m: tip error that counts as arrived
    approach_timeout_s=12.0,   # s: give up waiting and report where it got to
    approach_tip_left="",      # "x,y,z" world override of the working tip
    approach_tip_right="",     # (empty = the measured default)
    min_depth=0.10,            # m: hard insertion floor (the trocar must stay
    #                            threaded); commands shallower than this are
    #                            clamped in place, never accumulated
    cmd_leash_m=0.015,         # m: the tip command may never get further than
    #                            this from the tip the arm actually reached.
    #                            This is the anti-windup the human operator
    #                            provides in teleop (they feel the haptic wall
    #                            and stop pushing) and a policy does not. See
    #                            ArmRig.apply().
    rcm_abort_mm=10.0,         # trocar violation LEVEL that ENDS the episode.
    #                            Raised 3 -> 10 mm on 2026-08-13. 3 mm is the
    #                            port's own budget (a 6 mm hole around a 5 mm
    #                            shaft), and it is still what the SCORE is read
    #                            against -- see rcm_excursion_mm below. But it
    #                            was killing runs that live teleop survives
    #                            unremarked: teleop has no violation check at
    #                            all, and a two-arm session on this machine
    #                            (out/teleop_log.txt, 2026-08-13) reached a
    #                            worst RCM of 10.06 mm left / 17.58 mm right
    #                            with the shaft on the rubber for 11 % of ticks.
    #                            10 mm is not an arbitrary loosening: it is
    #                            --roll-gate-hi, the level at which the teleop
    #                            path declares the shaft to be LEAVING the port
    #                            and stops winding roll into it entirely
    #                            (teleop_left_touch.py, ARGS.roll_gate_hi).
    #                            THIS DOES NOT MAKE 10 mm SAFE. It is 7 mm
    #                            outside a 6 mm hole; on real tissue the port
    #                            is torn. It only stops the abort from being
    #                            the first thing a policy meets.
    rcm_excursion_mm=3.0,      # ...and the level the EXCURSION COUNTERS keep
    #                            using, deliberately left on the physical
    #                            budget. rcm_excursion_frac / _steps are what
    #                            the platform ranks port health on, so moving
    #                            the abort must not silently blind them: after
    #                            this change an episode can finish and still
    #                            report that it spent half its steps out of the
    #                            trocar. Judge on rcm_err_p95_mm,
    #                            rcm_excursion_frac and trocar_on_rubber_frac,
    #                            not on the absence of an abort
    rcm_abort_steps=5,         # ...and how many CONSECUTIVE control steps it
    #                            has to stay there before the episode is
    #                            terminated. A grasp that suddenly takes load
    #                            throws the tool for 2-3 steps and then
    #                            recovers (measured: 16.4 mm at load onset,
    #                            back to 0.14 mm by step 28); aborting on the
    #                            first tick would discard runs that are fine.
    #                            Every excursion is still counted and reported.
    manip_hz=60.0,             # rate to refresh the manipulability climb at.
    #                            0 = every tick (bit-exact laparo_control.BEST,
    #                            ~4x more CPU per control tick, same result)
    roll_follow="absolute",    # "absolute" = the roll command is a TARGET the
    #                            joint catches up to; "increment" = the old
    #                            lossy path, kept for A/B. See ArmRig.apply()
    gripper_rate_dps=250.0,    # included-angle slew rate of the jaws
    grasp_force=0.03,          # N: the force-limited close stops here
    jaw_length=0.015,          # m: blade lever arm, for the grip-force model
    grasp_radius=0.012,        # m: latch gate, tissue-in-mouth to grip handle
    grasp_assist=True,
    cut_through=True,          # the RIGHT cutter's blades drop their colliders
                               # while they open or close, so closing on the
                               # appendix sweeps through it instead of holding
                               # it (PhysX cannot re-mesh a cooked deformable,
                               # so it cannot be genuinely divided). Still
                               # blades collide, which keeps pushing and
                               # retracting tissue working. See ARM_CFG
    cut_dwell_s=0.15,          # s the blades must be still before the
                               # colliders come back
    cut_clear_m=0.004,         # m: ...and how far the blade segment must be
                               # from the appendix as well. A shut mouth reads
                               # as empty, so without this a close that ends
                               # inside the organ goes solid there
    # ---- the resection task: grasp the MID-BODY, lift, cut between the grasp
    # and the base. See _task_obs and vla_rollout.Episode.score
    cut_open_frac=0.50,        # blades must be at least this open, WITH
                               # appendix in the mouth, to arm a cut
    cut_shut_frac=0.15,        # ...and close past this to complete it
    grasp_band=(0.30, 0.70),   # the mid-body band, as a fraction along the
                               # appendix (0 = free tip, 1 = pinned base). A
                               # grasp outside it is not the requested one:
                               # the tip is no longer visible in the scope
                               # (user), and grabbing at the base leaves no
                               # appendix between the grasp and the body to
                               # cut through
    lift_success_m=0.010,      # m the appendix must rise while held
    resect_reach_m=0.005,      # m: how close the right tip must come to the
                               # resection segment to count as "at the site"
    resect_offset_m=0.010,     # m along the appendix from the GRASP toward the
                               # body: where the cut is meant to go. The scored
                               # stage still accepts anywhere in the segment
                               # between the grasp and the base; this is the
                               # point the scripted baseline aims at

    stereo=False,              # render the D405's second imager as well
    endo_depth=False,          # the scope has no depth sensor in reality
    endo_vignette=0.90,        # scope circle radius / half frame width. A rigid
                               # scope throws a CIRCLE on a rectangular sensor,
                               # so the corners of a real frame are black; 0
                               # disables and gives the old full rectangle.
                               # 0.75 threw away a third of the frame and put
                               # the working area behind the mask; 1.05 ran the
                               # circle past the left/right edges and barely
                               # read as a scope at all. 0.90 is the midpoint
                               # both complaints converged on
    endo_vignette_soft=0.02,   # edge falloff, same units (no scope cuts to
                               # black in one pixel)
    d405_baseline=0.018,       # m: the real D405 stereo baseline
    max_episode_steps=0,       # 0 = unbounded
    seed=0,
)

ARMS = ("left", "right")

ARM_CFG = {
    "left":  dict(urdf="out/fr3_left_grasper.urdf",
                  root="/World/robots/fr3_left",
                  other="/World/robots/fr3_right",
                  port_sign=-1.0,
                  joints=("gripper_jaw_left", "gripper_jaw_right"),
                  open_deg=37.0,               # PER JAW; included = 74 deg
                  endtip="jaw", endtip_mode="grasp", instrumented=True),
    "right": dict(urdf="out/fr3_right_cutter.urdf",
                  root="/World/robots/fr3_right",
                  other="/World/robots/fr3_left",
                  port_sign=+1.0,
                  joints=("cutter_blade_left", "cutter_blade_right"),
                  open_deg=25.0,               # PER JAW; included = 50 deg
                  endtip="blade", endtip_mode="cut", instrumented=False),
}
# endtip_mode: what the two endtip links do to tissue they close on. "grasp" =
# the jaws stop at the force floor and latch an appendix grip cluster; "cut" =
# the blades neither stop nor hold, and drop their colliders WHILE MOVING so
# they sweep through instead of dragging the organ. Kept identical to the
# teleop's ARM_CFG, which carries the full rationale -- the two must agree or a
# policy trained here meets a differently-behaved cutter on the rig.

ENDO_CAM = "/World/colliders/endoscope/endoscope_cam"
D405_CAM = "/World/sensors/realsense_d405/d405"
D405_ROOT = "/World/sensors/realsense_d405"

# jaw mouth, measured from the exported blade STLs (see teleop_left_touch.py)
JAW_BLADE_LEN = 0.021
JAW_HALF_WIDTH = 0.0025
JAW_SKIN = 0.0015

G_WORLD = np.array([0.0, 0.0, -9.81])

# The articulation force sensor reports the wrench the JOINT applies to its
# child link. For a quasi-static tool that is -(external load + weight), so
# once the weight is modelled out the remaining reaction has to be flipped to
# read as "what the tissue does to the instrument". Verified against a
# commanded dive into the organ bed (see VLA_HANDOFF.md, force validation).
REACTION_SIGN = -1.0


def _log(msg):
    print(msg, flush=True)


# ==========================================================================
# Isaac boot.  Importing this module does NOT start Isaac; `LaparoVLAEnv`
# does, on construction, because SimulationApp must exist before any omni
# import and its headless flag is a constructor argument.
# ==========================================================================
_APP = None
_APP_HEADLESS = None


def _boot(headless: bool):
    """Start (once) the Isaac SimulationApp and import everything omni-side."""
    global _APP, _APP_HEADLESS
    if _APP is not None:
        if _APP_HEADLESS != headless:
            raise RuntimeError(
                f"SimulationApp already running headless={_APP_HEADLESS}; one "
                f"process cannot host both. Build the env once per process.")
        return _APP
    from isaacsim import SimulationApp
    _APP = SimulationApp({"headless": headless,
                          "renderer": "RaytracedLighting"})
    _APP_HEADLESS = headless
    return _APP


# ==========================================================================
# small numeric helpers
# ==========================================================================
def _world_transform(prim):
    from pxr import Usd, UsdGeom
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    return np.array([[m[i][j] for j in range(4)] for i in range(4)]).T


def _rotvec_to_R(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, np.float64) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R_to_quat(R):
    """(w, x, y, z), the USD/Isaac ordering."""
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _T(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def _inv_T(M):
    R, p = M[:3, :3], M[:3, 3]
    return _T(R.T, -R.T @ p)


# ==========================================================================
# the action
# ==========================================================================
class Action:
    """Helpers for building the `(2, K, 7)` action array.

    The env accepts a raw ndarray as well; this is only sugar.
    """

    LEFT, RIGHT = 0, 1
    DX, DY, DZ, RX, RY, RZ, GRIP = range(7)

    @staticmethod
    def zeros(chunk: int, gripper=(0.0, 0.0)) -> np.ndarray:
        a = np.zeros((2, chunk, 7), np.float32)
        a[Action.LEFT, :, Action.GRIP] = gripper[0]
        a[Action.RIGHT, :, Action.GRIP] = gripper[1]
        return a

    @staticmethod
    def check(a, chunk: int) -> np.ndarray:
        """Validate and canonicalise. Accepts (2,K,7), (K,7) [left only paired
        with a held right] or a {'left':..., 'right':...} dict."""
        if isinstance(a, dict):
            per = {}
            for nm in ARMS:
                v = a.get(nm)
                if v is None:
                    continue
                v = np.atleast_2d(np.asarray(v, np.float32))
                if v.shape[1] != 7:
                    raise ValueError(
                        f"action['{nm}'] must be (K,7), got {v.shape}")
                per[nm] = v
            if not per:
                raise ValueError("action dict has neither 'left' nor 'right'")
            k = max(len(v) for v in per.values())
            out = np.zeros((2, k, 7), np.float32)
            for i, nm in enumerate(ARMS):
                if nm in per:
                    out[i, :len(per[nm])] = per[nm]
            a = out
        a = np.asarray(a, np.float32)
        if a.ndim == 2:                      # (K,7): both arms get it
            a = np.stack([a, a], 0)
        if a.ndim != 3 or a.shape[0] != 2 or a.shape[2] != 7:
            raise ValueError(f"action must be (2, K, 7), got {a.shape}")
        if a.shape[1] > chunk:
            raise ValueError(
                f"action chunk {a.shape[1]} exceeds the env's chunk={chunk}; "
                f"build the env with chunk >= your policy's horizon, or send "
                f"fewer sub-steps")
        if not np.all(np.isfinite(a)):
            raise ValueError("action contains NaN/Inf")
        return a


# ==========================================================================
# tissue in the jaw mouth  (verbatim from teleop_left_touch.tissue_probe)
# ==========================================================================
def tissue_probe(v_live, p_tip, a_shaft, R_flange, grip_half, jaw_pin_axis_f):
    if v_live is None:
        return False, None, None
    w = v_live - p_tip
    along = w @ (-a_shaft)
    n = R_flange @ jaw_pin_axis_f
    n = n - a_shaft * float(n @ a_shaft)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return False, None, None
    n = n / nn
    u = np.cross(a_shaft, n)
    s_u, s_n = w @ u, w @ n
    r = JAW_BLADE_LEN - along
    half_gap = r * math.sin(max(grip_half, 0.0))
    mouth = ((along > -JAW_SKIN) & (half_gap > JAW_SKIN)
             & (np.abs(s_u) < half_gap + JAW_SKIN)
             & (np.abs(s_n) < JAW_HALF_WIDTH + JAW_SKIN))
    if mouth.sum() < 5:
        return False, None, None
    su = s_u[mouth]
    if su.min() > JAW_SKIN or su.max() < -JAW_SKIN:
        return False, None, None
    thick = float(np.percentile(su, 95) - np.percentile(su, 5))
    return True, thick, v_live[mouth].mean(axis=0)


# ==========================================================================
# cameras
# ==========================================================================
class CamSensor:
    """One USD camera + its render product, with the intrinsics the RTX
    renderer actually uses (NOT the authored verticalAperture -- see
    start_pose.endoscope_camera for why they differ)."""

    def __init__(self, stage, prim_path, name, width, height, want_depth,
                 vignette=0.0, vignette_soft=0.02):
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom
        self.path = prim_path
        self.name = name
        self.W, self.H = int(width), int(height)
        self.want_depth = bool(want_depth)
        self._vig = self._vignette_mask(vignette, vignette_soft)
        self.cam = Camera(prim_path=prim_path, name=name,
                          resolution=(self.W, self.H))
        self.cam.initialize()
        self._apply_lens_distortion(stage, prim_path)
        if want_depth:
            self.cam.add_distance_to_image_plane_to_frame()
        prim = stage.GetPrimAtPath(prim_path)
        c = UsdGeom.Camera(prim)
        f = float(c.GetFocalLengthAttr().Get())
        hap = float(c.GetHorizontalApertureAttr().Get())
        hoff = float(c.GetHorizontalApertureOffsetAttr().Get() or 0.0)
        voff = float(c.GetVerticalApertureOffsetAttr().Get() or 0.0)
        px_per_ap = self.W / hap
        fx = fy = self.W * f / hap             # square pixels, as rendered
        cx = self.W / 2.0 - hoff * px_per_ap   # mirrored in u
        cy = self.H / 2.0 + voff * px_per_ap   # scaled by the HORIZONTAL ap
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                          np.float64)
        self.hfov = 2.0 * math.degrees(math.atan(self.W / (2.0 * fx)))
        self.vfov = 2.0 * math.degrees(math.atan(self.H / (2.0 * fy)))
        self._prim = prim
        self._last_rgb = None
        self._last_depth = None

    def _apply_lens_distortion(self, stage, prim_path):
        """Re-apply the stage's OpenCV lens distortion through the Camera API.

        `build_surgical_cell` authors `omni:lensdistortion:*` on the camera prim,
        but authoring the attributes is not by itself enough to make RTX render
        the barrel -- measured: an A/B of k1 = -0.20 against k1 = 0 came back
        with a mean pixel difference of 1.09 out of 255, i.e. render noise, in
        both the viewport capture and this annotator path. The Camera wrapper
        does more than write attributes, so the values are pushed back through
        it here, where the render product exists.
        """
        prim = stage.GetPrimAtPath(prim_path)
        a = prim.GetAttribute("omni:lensdistortion:opencvPinhole:k1")
        if not (a and a.Get()):
            return
        pre = "omni:lensdistortion:opencvPinhole:"

        def val(n, d=0.0):
            at = prim.GetAttribute(pre + n)
            return float(at.Get()) if at and at.Get() is not None else d

        try:
            self.cam.set_opencv_pinhole_properties(
                cx=val("cx", self.W / 2), cy=val("cy", self.H / 2),
                fx=val("fx"), fy=val("fy"),
                pinhole=[val("k1"), val("k2"), val("p1"), val("p2"),
                         val("k3"), val("k4"), val("k5"), val("k6"),
                         val("s1"), val("s2"), val("s3"), val("s4")])
            _log(f"[{self.name}] opencvPinhole distortion applied: "
                 f"k1={val('k1'):+.3f}")
        except Exception as exc:                      # noqa: BLE001
            _log(f"[{self.name}] lens distortion NOT applied: {exc}")

    def _vignette_mask(self, frac, soft):
        """The circular field a rigid scope actually projects, or None.

        A laparoscope's objective throws a CIRCLE onto a rectangular sensor, so
        the corners of a real scope frame carry no image at all -- they are
        black. The sim camera fills the whole rectangle, which is the single
        most obvious tell when a sim frame is put beside a real one, and a
        policy trained on full-rectangle frames sees image where the real
        scope will hand it none.

        `frac` is the circle radius as a fraction of HALF THE FRAME WIDTH, so
        1.0 makes the circle touch the left and right edges (the usual fit on a
        16:9 tower, which then crops the circle top and bottom). `soft` is the
        edge falloff in the same units -- real scopes do not cut to black on
        one pixel.
        """
        if not frac or frac <= 0.0:
            return None
        r_px = 0.5 * self.W * float(frac)
        soft_px = max(1e-6, 0.5 * self.W * float(soft))
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        d = np.hypot(xx - (self.W - 1) / 2.0, yy - (self.H - 1) / 2.0)
        m = np.clip((r_px - d) / soft_px, 0.0, 1.0)
        return m.astype(np.float32)[..., None]

    def _apply_vignette(self, rgb):
        if self._vig is None:
            return rgb
        return (rgb.astype(np.float32) * self._vig).astype(np.uint8)

    def pose_world(self):
        """4x4 world <- camera, in the OPENCV convention (+x right, +y DOWN,
        +z forward). USD cameras are +y up / -z forward, so two axes flip."""
        T = _world_transform(self._prim)
        R = T[:3, :3]
        R = R / np.linalg.norm(R, axis=0, keepdims=True)
        Rcv = np.column_stack([R[:, 0], -R[:, 1], -R[:, 2]])
        return _T(Rcv, T[:3, 3])

    def read(self):
        """Always returns the keys it advertises.

        The RTX annotators do not publish on the first render of a run, so a
        naive read returns nothing and `obs['endoscope']['rgb']` raises
        KeyError on the very first observation -- which is a hostile interface,
        and exactly what a policy hits on its first call. A real sensor holds
        its last value instead, so this one does too: last good frame, or
        zeros if there has never been one, with `stale` saying which."""
        out = {}
        rgb = self.cam.get_rgb()
        if rgb is not None and np.size(rgb):
            rgb = np.asarray(rgb)
            self._last_rgb = self._apply_vignette(
                (rgb[..., :3] if rgb.ndim == 3
                 and rgb.shape[2] >= 3 else rgb).astype(np.uint8))
            out["stale"] = False
        else:
            out["stale"] = True
        out["rgb"] = (self._last_rgb if self._last_rgb is not None
                      else np.zeros((self.H, self.W, 3), np.uint8))
        if self.want_depth:
            d = self.cam.get_depth()
            if d is not None and np.size(d):
                d = np.asarray(d, np.float32).copy()
                d[~np.isfinite(d)] = 0.0
                self._last_depth = d
            out["depth"] = (self._last_depth if self._last_depth is not None
                            else np.zeros((self.H, self.W), np.float32))
        return out

    def info(self):
        return dict(K=self.K.copy(), width=self.W, height=self.H,
                    hfov_deg=self.hfov, vfov_deg=self.vfov)


# ==========================================================================
# one arm
# ==========================================================================
class ArmRig:
    def __init__(self, name, env, seed_q=None):
        from laparo_ik import Chain, port_inward_normal, rcm_manipulability
        from start_pose import solve_start_q, SEED
        from pxr import Usd, UsdPhysics

        self.name = name
        self.env = env
        self.cfg = ARM_CFG[name]
        stage = env.stage
        self.stage = stage

        grp = stage.GetPrimAtPath(self.cfg["root"])
        if not grp:
            raise RuntimeError(f"no robot at {self.cfg['root']}")
        self.grp = grp
        self.T_base = _world_transform(grp)

        self.p_rcm, self.port_no = None, None
        for i in (2, 3):
            prim = stage.GetPrimAtPath(f"/World/trocarPorts/port_{i:02d}")
            if prim:
                p = _world_transform(prim)[:3, 3]
                if p[0] * self.cfg["port_sign"] > 0:
                    self.p_rcm, self.port_no = p, i
        if self.p_rcm is None:
            raise RuntimeError(f"no {name} trocar port marker found")

        self.chain = Chain(ROOT / self.cfg["urdf"])
        self.u_in = env.R_ports @ port_inward_normal(self.port_no)
        # see ARM_CFG: only a grasper senses tissue in its mouth, stops its
        # close on it and latches it
        self.grips = self.cfg.get("endtip_mode", "grasp") == "grasp"

        self.p_start = env.start_tips.get(name)
        if self.p_start is None:
            self.p_start = self.p_rcm + env.cfg["insert"] * self.u_in

        cfgc = env.control_cfg
        self.ik_kw = dict(k_manip=cfgc.k_manip, k_limit=cfgc.k_limit,
                          limit_margin=math.radians(cfgc.limit_margin_deg))
        q, sinfo = solve_start_q(
            self.chain, self.T_base, self.p_start, self.p_rcm,
            seed=SEED if seed_q is None else seed_q, roll_deg=0.0,
            elbow_lift=0.6, elbow_point=3, ik_kw=self.ik_kw)
        self.q0 = q.copy()
        self.q = q.copy()
        self.r_cmd0 = sinfo["r_cmd"]
        _log(f"[{name:>5}] start pose: tip err {sinfo['tip_err']*1000:.2f} mm, "
             f"RCM err {sinfo['rcm_err']*1000:.2f} mm, manip "
             f"{rcm_manipulability(self.chain, q, self.T_base, self.p_rcm):.1e}")

        # TOOL <- tool_flange body frame, so a wrench reported by the joint
        # sensor lands directly on the URDF tool_tip axes (see tool_axes()):
        #   chain.R_f2t maps TIP -> FLANGE, so its transpose is FLANGE -> TIP
        #   steps[8] is the fixed fr3_link8_to_tool joint, i.e. link8 <- tool_flange
        self.R_tool_from_flange = self.chain.R_f2t.T.copy()
        self.R_fl_from_toolflange = self.chain.steps[8][0][:3, :3].copy()
        self.R_tool_from_toolflange = (self.R_tool_from_flange
                                       @ self.R_fl_from_toolflange)
        self.grip_half_max = math.radians(self.cfg["open_deg"])
        self.art = None
        self.wrench_model = None          # identified once, cached to disk
        self._csamp, self._jsamp = [], []
        self._wacc, self._jacc, self._wn, self._jaw_tau = np.zeros(6), 0.0, 0, 0.0
        self.wrench_tool = np.zeros(6)
        self.w_tare = np.zeros(6)

    # ---- physics binding -------------------------------------------------
    def attach(self):
        from isaacsim.core.prims import SingleArticulation
        from pxr import Usd, UsdPhysics
        art_root = None
        for prim in Usd.PrimRange(self.grp):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_root = str(prim.GetPath())
                break
        if art_root is None:
            raise RuntimeError(f"no articulation root under {self.cfg['root']}")
        self.art = SingleArticulation(art_root)
        self.art.initialize()
        self.dof_names = list(self.art.dof_names)
        self.idx = [self.dof_names.index(f"fr3_joint{i}") for i in range(1, 8)]
        self.ctrl = self.art.get_articulation_controller()
        cfgc = self.env.control_cfg
        self.ctrl.set_gains(kps=np.full(len(self.dof_names), cfgc.kp),
                            kds=np.full(len(self.dof_names), cfgc.kd))
        self.jidx = [self.dof_names.index(n) for n in self.cfg["joints"]
                     if n in self.dof_names]
        if len(self.jidx) != 2:
            self.jidx = []
            _log(f"[{self.name:>5}] WARNING: no {self.cfg['endtip']} dofs "
                 f"({self.cfg['joints']}) -- gripper channel is inert")
        self._map_force_rows()
        self._find_blade_colliders()

    def _find_blade_colliders(self):
        """The endtip links' colliders, as collisionEnabled handles -- the
        teleop's `_find_blade_colliders`, on this env's stage. "cut" arms only:
        a grasper's jaws stay solid at all times."""
        from pxr import Usd, UsdPhysics
        self.blade_en = []
        self.blade_through = False
        self.blade_still_s = 0.0
        if self.grips or not self.env.cfg["cut_through"]:
            return
        # Visible only because build_surgical_cell de-instanced the endtip
        # `collisions` scopes; de-instancing here would change composition
        # after physics parsed the stage and kill the tensor view.
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.cfg["root"])):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = prim.GetPath().pathString
            if any(f"/{self.cfg['endtip']}_{side}/" in path
                   for side in ("left", "right")):
                self.blade_en.append(
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr())
        _log(f"[{self.name:>5}] cut-through: {len(self.blade_en)} blade "
             f"collider(s) drop out while the blades move "
             f"(dwell {self.env.cfg['cut_dwell_s']*1000:.0f} ms)"
             + ("" if self.blade_en else
                " -- NONE editable, rebuild out/surgical_cell.usd"))

    def _blade_in_tissue(self):
        """Is either blade still buried in the appendix? Distance from the live
        surface to the blade SEGMENT (pin -> tip), not to the tip point."""
        v = self.env.live_appendix_points()
        if v is None or not len(v):
            return False
        b = self.p_tip_m - self.a * JAW_BLADE_LEN
        d = self.p_tip_m - b
        t = np.clip(((v - b) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
        return bool(np.linalg.norm(v - (b + t[:, None] * d),
                                   axis=1).min() < self.env.cfg["cut_clear_m"])

    def _step_blade_phase(self, moved, dt):
        """Ghost the blades while they work tissue; solid again only once they
        are still, empty-mouthed AND clear of it. See the teleop's copy for why
        the clearance test is needed -- a SHUT mouth reads as empty, so without
        it a close that ends inside the appendix goes solid there and the organ
        sticks to the cutter."""
        if not self.blade_en:
            return
        working = moved or self.tissue_near
        self.blade_still_s = 0.0 if working else self.blade_still_s + dt
        if self.blade_through:
            want = (working or self.blade_still_s < self.env.cfg["cut_dwell_s"]
                    or self._blade_in_tissue())
        else:
            want = working
        if want == self.blade_through:
            return
        for attr in self.blade_en:
            attr.Set(not want)
        self.blade_through = want

    def _map_force_rows(self):
        """Row indices into get_measured_joint_forces() for the tool mount and
        the two jaw joints. Row = joint_index + 1 (row 0 is the base link)."""
        self.row_tool = None
        self.row_jaws = []
        try:
            md = self.art._articulation_view._metadata
            ji = md.joint_indices
            for cand in ("fr3_link8_to_tool", "tool_flange_to_tip"):
                if cand in ji:
                    self.row_tool = int(ji[cand]) + 1
                    self.tool_joint = cand
                    break
            for n in self.cfg["joints"]:
                if n in ji:
                    self.row_jaws.append(int(ji[n]) + 1)
            self.joint_names_meta = list(md.joint_names)
            _log(f"[{self.name:>5}] force sensor: joint order "
                 f"{self.joint_names_meta}")
            _log(f"[{self.name:>5}] force sensor: tool mount "
                 f"'{getattr(self, 'tool_joint', None)}' -> row {self.row_tool}"
                 f", jaws -> rows {self.row_jaws}")
        except Exception as exc:               # pragma: no cover
            _log(f"[{self.name:>5}] joint-force metadata unavailable ({exc}) "
                 f"-- the wrench channel falls back to the model estimate")
            self.joint_names_meta = []

    # ---- runtime state ---------------------------------------------------
    def reset_state(self):
        from laparo_control import TipController
        env = self.env
        full = self.art.get_joint_positions()
        full[self.idx] = self.q0
        if self.jidx:
            full[self.jidx] = 0.0
        self.art.set_joint_positions(full)
        self.art.set_joint_velocities(np.zeros_like(full))
        self.q = self.q0.copy()
        self.p_cmd = self.p_start.copy()
        self.p_cmd_prev = self.p_cmd.copy()
        self.p_shaped = self.p_cmd.copy()
        self.v_cmd = np.zeros(3)
        self.r_cmd = None if self.r_cmd0 is None else self.r_cmd0.copy()
        self.con = TipController(env.control_cfg, self.chain, self.T_base,
                                 self.p_rcm, self.q0,
                                 min_depth=env.cfg["min_depth"],
                                 port_normal=self.u_in)
        self.grip_half = 0.0
        self.grip_cmd_frac = 0.0
        self.grip_hold = 0.0
        self.grip_meas = np.zeros(2)
        self.grasp_thick = None
        self.grasped = False
        self.grasp_cluster = None
        self.follow_h0 = np.zeros(3)
        self.follow_t0 = np.zeros(3)
        self.tissue_near = False
        self.floor_half = 0.0
        self.probe_c = None
        self.probe_thick = 0.0
        # the cut gesture (right arm only) -- see _step_cut
        self.cut_armed = False
        self.cut_done = False
        self.cut_site = None
        self.cut_t = float("nan")
        self.n_cuts = 0
        self.obs_segs = []
        self.droll_req = 0.0
        self.roll_ach = 0.0
        self.roll_cmd = 0.0        # accumulated shaft roll the action asked for
        self.roll_put = 0.0        # accumulated roll joint 7 actually took
        self.depth = env.cfg["insert"]
        self.rcm_err = 0.0
        self.bite = 0.0
        self.lag = np.zeros(3)
        self.n_depth_clamped = 0
        self.n_cone_cmd = 0
        self.n_leash = 0
        self.rot_rejected = np.zeros(3)
        self.wrench_tool = np.zeros(6)
        self.grip_force = 0.0
        self.grip_force_raw = 0.0
        self._csamp, self._jsamp = [], []
        self._grip_prev = 0.0
        self._wacc, self._jacc, self._wn, self._jaw_tau = np.zeros(6), 0.0, 0, 0.0
        # NOT reset here: the tare is a property of the start pose, and reset()
        # re-measures it after the settle
        # measured, filled by refresh_measured()
        self.qs = self.q0.copy()
        self.qd = np.zeros(7)
        self.refresh_measured()
        # aperture cone
        cfgc = env.control_cfg
        import trocar
        self.tilt_max = trocar.clamp_tilt_rad(cfgc.cone_mm, cfgc.tilt_cap_deg)

    def refresh_measured(self):
        from laparo_ik import SHAFT_AXIS_F
        self.qs = self.art.get_joint_positions()[self.idx]
        v = self.art.get_joint_velocities()
        self.qd = v[self.idx] if v is not None else np.zeros(7)
        self.p_tip_m, self.R_fl = self.chain.fk(self.qs, self.T_base)
        self.a = self.R_fl @ SHAFT_AXIS_F
        if self.jidx:
            self.grip_meas = self.art.get_joint_positions()[self.jidx]

    # ---- the force channel -----------------------------------------------
    #
    # The articulation reports the wrench each joint applies to its child link.
    # At the tool mount that is
    #
    #     w_meas(R) = -( external load ) + payload(R) + sensor bias
    #
    # where `payload(R)` is the instrument's own weight seen in the (rotating)
    # tool_flange frame. Subtracting an ANALYTIC payload from the USD masses
    # does not work: the residual came out at 4.2 N -- as large as the weight
    # itself -- and, being a mis-modelled ROTATING term treated as a constant,
    # it drifted 0.8 N across 19 deg of shaft roll and would have read as
    # tissue contact. So the payload is IDENTIFIED instead, exactly the way a
    # real wrist F/T sensor is calibrated against its tool:
    #
    #     f_b(R) = R^T V + b_f                    V = the world-frame weight
    #     t_b(R) = c x (R^T V) + b_t              c = the payload COM
    #
    # both linear in the unknowns, fitted by least squares over a free-space
    # excitation (shaft roll + a small tip Lissajous) at reset. `fit_rms_*`
    # is the residual of that fit and IS the noise floor of the channel --
    # anything smaller than it is not a measurement.
    #
    def calib_sample(self, wrench=True, jaw=False):
        """One (orientation, wrench, joint speed) and/or (opening, torque)
        sample.

        The joint speed rides along because the model is STATIC: a sample taken
        while the arm is still moving measures the excitation, not the payload.
        Rather than gate here on a threshold guessed in advance (a first
        attempt did, and threw away all 100 % of its samples), every sample is
        kept and the fit uses the quietest ones.

        `wrench` and `jaw` are separate because the jaw drives run at the same
        5e5 stiffness as the arm on a 4 g link: sweeping the gripper while
        identifying the PAYLOAD injects the jaw drives' own reaction into the
        tool-mount samples."""
        if wrench:
            w = self._raw_wrench()
            if w is not None:
                R = self.R_fl @ self.R_fl_from_toolflange   # world <- tool_flange
                self._csamp.append((R.copy(), w.copy(),
                                    float(np.abs(self.qd).max())))
        if jaw:
            t = self._raw_jaw_torque()
            if t is not None:
                self._jsamp.append((float(self.grip_half), float(t)))

    def fit_wrench_model(self):
        """Least-squares identification of the payload + bias model."""
        if len(self._csamp) < 30:
            _log(f"[{self.name:>5}] wrench calibration SKIPPED "
                 f"({len(self._csamp)} samples) -- the force channel will "
                 f"report zero")
            return None
        qd = np.array([s[2] for s in self._csamp])
        keep = qd <= max(np.percentile(qd, 60), 1e-6)
        samp = [s for s, k in zip(self._csamp, keep) if k]
        A, y = [], []
        for R, w, _ in samp:
            A.append(np.hstack([R.T, np.eye(3)]))
            y.append(w[:3])
        u, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(y), rcond=None)
        V, b_f = u[:3], u[3:]
        B, z = [], []
        for R, w, _ in samp:
            g = R.T @ V
            gx = np.array([[0, -g[2], g[1]], [g[2], 0, -g[0]], [-g[1], g[0], 0]])
            B.append(np.hstack([-gx, np.eye(3)]))
            z.append(w[3:])
        v, *_ = np.linalg.lstsq(np.vstack(B), np.concatenate(z), rcond=None)
        c, b_t = v[:3], v[3:]
        rf = np.array([w[:3] - (R.T @ V + b_f) for R, w, _ in samp])
        rt = np.array([w[3:] - (np.cross(c, R.T @ V) + b_t)
                       for R, w, _ in samp])
        # jaw baseline: the free-space reaction as a function of the opening
        # angle, taken from a separate sweep with the arm held still
        ja, jb, jr = 0.0, 0.0, 0.0
        if len(self._jsamp) >= 8:
            th = np.array([s[0] for s in self._jsamp])
            tq = np.array([s[1] for s in self._jsamp])
            M = np.column_stack([th, np.ones_like(th)])
            (ja, jb), *_ = np.linalg.lstsq(M, tq, rcond=None)
            jr = float(np.sqrt(np.mean((tq - M @ [ja, jb]) ** 2)))
        self.wrench_model = dict(
            V=V.tolist(), b_f=b_f.tolist(), c=c.tolist(), b_t=b_t.tolist(),
            jaw_a=float(ja), jaw_b=float(jb),
            rms_f=float(np.sqrt(np.mean(np.sum(rf ** 2, 1)))),
            rms_t=float(np.sqrt(np.mean(np.sum(rt ** 2, 1)))),
            rms_jaw=jr, n=len(samp), n_raw=len(self._csamp),
            n_jaw=len(self._jsamp))
        m = self.wrench_model
        _log(f"[{self.name:>5}] wrench model identified from {m['n']} poses: "
             f"payload {np.linalg.norm(V):.2f} N at COM "
             f"{np.round(np.array(c)*1000,1).tolist()} mm; residual "
             f"{m['rms_f']*1000:.1f} mN / {m['rms_t']*1000:.2f} mN.m; "
             f"jaw baseline residual {jr*1e6:.1f} uN.m over {m['n_jaw']} "
             f"openings")
        self._csamp, self._jsamp = [], []
        return self.wrench_model

    def _raw_wrench(self):
        if self.row_tool is None:
            return None
        try:
            f = np.asarray(self.art.get_measured_joint_forces(), np.float64)
        except Exception:
            return None
        if f is None or self.row_tool >= len(f):
            return None
        return f[self.row_tool].copy()

    def _raw_jaw_torque(self):
        """Mean magnitude of the reaction torque at the two jaw joints, N.m."""
        if not self.row_jaws:
            return None
        try:
            m = np.asarray(self.art.get_measured_joint_forces(), np.float64)
        except Exception:
            return None
        tau = [float(np.linalg.norm(m[r][3:])) for r in self.row_jaws
               if r < len(m)]
        return float(np.mean(tau)) if tau else None

    def wrench_tool_now(self):
        """This tick's payload-compensated wrench, in the TOOL frame, BEFORE
        the tare. None if the sensor or the model is unavailable."""
        w = self._raw_wrench()
        if w is None or self.wrench_model is None:
            return None
        m = self.wrench_model
        R = self.R_fl @ self.R_fl_from_toolflange
        V, c = np.asarray(m["V"]), np.asarray(m["c"])
        g = R.T @ V
        w_ext = REACTION_SIGN * (w - np.concatenate(
            [g + np.asarray(m["b_f"]),
             np.cross(c, g) + np.asarray(m["b_t"])]))
        Rt = self.R_tool_from_toolflange
        return np.concatenate([Rt @ w_ext[:3], Rt @ w_ext[3:]])

    def sample_wrench(self):
        """One physics tick's external wrench, into the averaging accumulator.

        The channel is AVERAGED over the control step rather than sampled at
        its end. The articulation solver's tick-to-tick reaction noise is what
        sets the noise floor here, and averaging 16 ticks takes ~4x off it --
        which is also what the anti-alias filter on a real load cell does."""
        w = self.wrench_tool_now()
        if w is None:
            return
        self._wacc += w - self.w_tare
        t = self._raw_jaw_torque()
        if t is not None:
            self._jacc += t
        self._wn += 1

    def tare(self, samples):
        """Zero the channel at the pose an episode starts from.

        The payload model is identified 25 mm clear of the organ bed, so at the
        start pose it leaves a constant offset -- measured at 0.7 N, which is
        the size of the contacts the channel exists to report. Taring at the
        start pose removes it, exactly as an instrumented tool is tared before
        it goes through the port. It does NOT fix pose dependence; what is left
        of that is the `noise_force_n` the observation reports."""
        if samples:
            self.w_tare = self.w_tare + np.mean(samples, axis=0)
            _log(f"[{self.name:>5}] force channel tared at the start pose: "
                 f"removed {np.linalg.norm(np.mean(samples, axis=0)[:3]):.3f} N"
                 f" / {np.linalg.norm(np.mean(samples, axis=0)[3:]):.4f} N.m")

    def read_wrench(self):
        """External wrench APPLIED BY THE TISSUE to the instrument, averaged
        over the control step and resolved in the URDF tool_tip frame:
        (fx, fy, fz, tx, ty, tz), +x the jaw CLOSING direction, +y the clevis
        PIN axis, +z back along the shaft OUT of the patient."""
        if self._wn:
            self.wrench_tool = self._wacc / self._wn
            self._jaw_tau = self._jacc / self._wn
        self._wacc, self._jacc, self._wn = np.zeros(6), 0.0, 0
        return self.wrench_tool

    def read_grip_force(self):
        """Jaw closing force at the blade, N.

        Two channels, because they answer different questions:

        `grip_force` (reported as obs['left_force']['grip']) is the STRAIN law
        that actually governs the jaws in this simulator: the force-limited
        close stops where this reaches `grasp_force`, so it is the quantity
        that decides whether the tissue is pinched or crushed. It is a model,
        and it saturates at 75 mN by construction.

        `grip_force_raw` is the jaw joints' own reaction torque over the blade
        lever arm. It is a direct measurement, but the jaws are hard-set
        kinematically every tick and a latched grasp is a rigid constraint, so
        once `grasped` is true it reports the constraint force (tens of N) and
        not a bite force. Use it for contact DETECTION, not for magnitude."""
        env = self.env
        self.grip_force = 0.0
        if self.tissue_near and self.grasp_thick:
            gap = 2.0 * env.cfg["jaw_length"] * math.sin(self.grip_half)
            strain = float(np.clip(1.0 - gap / max(self.grasp_thick, 1e-6),
                                   0.0, 1.0))
            self.grip_force = 0.075 * strain
        elif self.grasped:
            self.grip_force = 0.075 * 0.4        # held at the settled floor
        self.grip_force_raw = 0.0
        if self.wrench_model is not None:
            m = self.wrench_model
            base = m["jaw_a"] * self.grip_half + m["jaw_b"]
            self.grip_force_raw = max(
                0.0, self._jaw_tau - base) / max(env.cfg["jaw_length"], 1e-6)
        return self.grip_force

    # ---- the frames the observation is built from ------------------------
    def tool_axes(self):
        """(x, y, z) unit columns of the URDF `tool_tip` frame, in world.

        NOT a frame invented here: it is the tool_tip link the instrument URDF
        actually defines (`tool_flange_to_tip`, rpy = pi, 0, 3pi/4), which is
        the frame the teleop reasons in -- so a policy trained on teleop
        demonstrations and a policy evaluated here speak the same axes.
        Identical on both arms despite the cutter's meshes being turned 180 deg
        about Z. In flange coordinates:

            tip +x = ( -0.707,  0.707,  0 )   the jaw CLOSING direction
            tip +y = (  0.707,  0.707,  0 )   the clevis PIN axis
            tip +z = (  0,      0,     -1 )   BACK along the shaft, out of the
                                              patient -- so insertion is -z

        The -z-points-forward convention is deliberate and shared with the
        OpenHaptics stylus: pushing the pen along its -z inserts the instrument
        along the shaft. `self.a` remains the physical insertion direction
        (= -tip z) and is what the RCM, depth and jaw-mouth maths use."""
        R = self.R_fl @ self.chain.R_f2t
        return R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()

    def rcm4(self):
        """The four DOF the trocar leaves: insertion, tilt, azimuth, roll.

        insertion  m, tip distance past the port centre along the shaft
        tilt       rad, shaft axis away from the port's inward surface normal
        azimuth    rad, which way it leans, about that normal, measured from
                   world +x projected into the port's tangent plane
        roll       rad, signed rotation of the tip frame's +y (the clevis pin
                   axis) about the tip frame's +z, from world +z projected
                   perpendicular to the shaft. Tracking tip +y against world up
                   is the same quantity `start_pose.solve_start_q` reports as
                   `y_up_deg`, so the roll here and the roll the start pose was
                   solved for are the same number.

        The shaft quantities use `self.a` (the INSERTION direction) rather than
        the tip frame's +z, which points the other way -- see tool_axes().
        """
        x, y, z = self.tool_axes()
        a = self.a / np.linalg.norm(self.a)          # insertion direction
        n = self.u_in / np.linalg.norm(self.u_in)
        ins = float(np.dot(self.p_tip_m - self.p_rcm, a))
        tilt = math.acos(float(np.clip(np.dot(a, n), -1.0, 1.0)))
        e1 = np.array([1.0, 0.0, 0.0]) - n * n[0]
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.array([0.0, 1.0, 0.0]) - n * n[1]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        lat = a - n * float(np.dot(a, n))
        az = (math.atan2(float(np.dot(lat, e2)), float(np.dot(lat, e1)))
              if np.linalg.norm(lat) > 1e-9 else 0.0)
        up = np.array([0.0, 0.0, 1.0])
        up = up - z * float(np.dot(up, z))
        if np.linalg.norm(up) < 1e-6:
            up = np.array([0.0, 1.0, 0.0]) - z * z[1]
        up /= np.linalg.norm(up)
        roll = math.atan2(float(np.dot(np.cross(up, y), z)),
                          float(np.dot(up, y)))
        return np.array([ins, tilt, az, roll])

    # ---- one physics tick ------------------------------------------------
    def apply(self, d_xyz_tool, d_rot_tool, grip_frac, dt, R_tool_step):
        """Advance the tip command by one physics tick's share of the action.

        `R_tool_step` is the tool frame captured at the start of the current
        action sub-step -- frozen for its duration so the commanded path is a
        straight line in that frame rather than a curve that chases its own
        output."""
        import trocar
        from laparo_ik import perp_unit, rodrigues, roll_error
        env = self.env

        self.p_cmd = self.p_cmd + R_tool_step @ np.asarray(d_xyz_tool, np.float64)
        # keep the command inside what the tool can physically deliver, IN
        # PLACE: an unreachable component must never accumulate across steps
        if env.cfg["min_depth"] > 0.0:
            v = self.p_cmd - self.p_rcm
            d = float(np.linalg.norm(v))
            if 1e-6 < d < env.cfg["min_depth"]:
                self.p_cmd = self.p_rcm + v / d * env.cfg["min_depth"]
                self.n_depth_clamped += 1
        if self.tilt_max > 0.0:
            self.p_cmd, hit = trocar.cone_clamp(self.p_cmd, self.p_rcm,
                                                self.u_in, self.tilt_max)
            self.n_cone_cmd += int(hit)
        # LEASH. Everything above bounds where the command may point; this
        # bounds how far AHEAD of the arm it may get, and it is the one that
        # matters for a policy.
        #
        # The tip command is an integrator of the policy's deltas. A policy
        # that closes its own loop on the measured tip -- "move toward the
        # target" -- keeps adding deltas for as long as the tip is short of
        # the target, so if the target is unreachable the command runs away
        # without bound. Measured on the scripted lift baseline: p_cmd ended
        # 104 mm ahead of the tip, and because the shipped controller trades
        # RCM against tip error (rcm_mode="soft", the grid-search winner), the
        # shaft came 86 mm out of a 6 mm hole. In teleop this cannot happen:
        # the operator feels the 3 N haptic wall and stops pushing. The leash
        # is that wall.
        leash = self.env.cfg["cmd_leash_m"]
        if leash > 0.0:
            d = self.p_cmd - self.p_tip_m
            n = float(np.linalg.norm(d))
            if n > leash:
                self.p_cmd = self.p_tip_m + d * (leash / n)
                self.n_leash += 1

        # ---- shaft roll: an absolute TARGET, not a per-tick increment ------
        #
        # Only the roll about the shaft survives the RCM; the other two
        # rotation axes are reported back as rejected.
        #
        # The roll used to be handed to the controller as the increment for
        # this tick, and `direct_roll` clips that to roll_qd_max * dt =
        # 0.30 deg per physics tick. Whatever the action asked for beyond that
        # was DISCARDED -- there was nothing left to retry, because the request
        # was consumed on read. The ceiling was n_sub * roll_qd_max * dt_phys =
        # 4.77 deg per 1/30 s action sub-step, and everything past it vanished
        # silently: unlike the two rejected rotation axes this loss was counted
        # nowhere, and a policy with no hand to feel it would simply learn to
        # over-command. (TELEOP_RESPONSIVENESS.md section 5.)
        #
        # Accumulating the request into a target and asking for the RESIDUAL
        # makes the roll behave like the position command, which is absolute
        # and rebuilt every tick: a rate clip becomes LAG instead of LOSS, and
        # the shaft always ends up where the action said, however fast it got
        # there. `roll_backlog` reports whatever is still owed.
        # `rz` is about the TIP frame's +z, which points OUT of the patient,
        # while joint 7 (and `self.a`) turn about the shaft pointing IN. They
        # are the same axis with opposite sign, so the request is negated once,
        # here, and everything downstream stays in shaft coordinates. The
        # accumulators below are kept in the ACTION's sign so that
        # roll_cmd/roll_meas/roll_backlog can be compared to what was asked.
        droll = float(d_rot_tool[2])
        self.rot_rejected = np.array([d_rot_tool[0], d_rot_tool[1], 0.0])
        self.roll_cmd += droll
        if env.cfg["roll_follow"] == "absolute":
            j = self.con.ROLL_JOINT
            resid = -(self.roll_cmd - self.roll_put)      # -> shaft sign
            # anti-windup at the joint STOP: never carry more residual than
            # joint 7 has travel left, and hand back what it cannot take --
            # otherwise un-twisting is dead until the debt clears
            buf = 1e-3
            room_hi = (self.chain.hi[j] - buf) - self.con.q[j]
            room_lo = (self.chain.lo[j] + buf) - self.con.q[j]
            resid = float(np.clip(resid, min(room_lo, 0.0), max(room_hi, 0.0)))
            self.roll_cmd = self.roll_put - resid        # back to action sign
            self.droll_req = resid
        else:
            self.droll_req = -droll         # the old lossy increment, for A/B
        # Advance the roll TARGET by the REQUEST, never by the residual, and do
        # it BEFORE the controller step. By the residual it overshoots 1.27x,
        # because the residuals summed over the catch-up ticks far exceed the
        # roll actually asked for; after the step, the IK's roll level still
        # sees the old target and can cancel the direct drive inside the same
        # tick (roll_cap is 4x the direct step).
        if droll != 0.0 and self.r_cmd is not None:
            self.r_cmd = rodrigues(self.a, -droll) @ self.r_cmd

        self._step_gripper(grip_frac, dt)
        self._step_grasp(grip_frac)

        if self.r_cmd is not None:
            self.r_cmd = perp_unit(self.r_cmd, self.a)
            from laparo_ik import JAW_PIN_AXIS_F
            phi = roll_error(self.a, self.R_fl @ JAW_PIN_AXIS_F, self.r_cmd)
            cap = math.radians(45.0)
            if abs(phi) > cap:
                self.r_cmd = perp_unit(
                    rodrigues(self.a, math.copysign(cap, phi) - phi)
                    @ self.r_cmd, self.a)

        self.depth = float(np.dot(self.p_tip_m - self.p_rcm, self.a))
        w_rcm = float(np.clip(self.depth / 0.02, 0.0, 1.0))

        g_obs = None
        if env.k_avoid > 0.0 and self.obs_segs:
            from laparo_ik import obstacle_gradient
            g_obs, _ = obstacle_gradient(self.chain, self.q, self.T_base,
                                         self.obs_segs, env.k_avoid,
                                         env.avoid_dist, env.avoid_radius)

        v_raw = (self.p_cmd - self.p_cmd_prev) / dt
        b = 1.0 - math.exp(-2.0 * math.pi * 8.0 * dt)
        self.v_cmd = self.v_cmd + (v_raw - self.v_cmd) * b
        self.p_cmd_prev = self.p_cmd.copy()

        q7_pre = float(self.con.q[self.con.ROLL_JOINT])
        self.q, qd_ff, self.p_shaped = self.con.step(
            self.p_cmd, self.v_cmd, self.p_tip_m, dt, w_rcm=w_rcm,
            r_cmd=self.r_cmd, g_extra=g_obs, line_meas=(self.p_tip_m, self.a),
            q_meas=self.qs, droll=self.droll_req)
        # what the roll joint ACTUALLY took this tick -- the accumulator has to
        # count the delivery, not the request, or the residual never clears and
        # the target runs away from the shaft. Negated back into the ACTION's
        # sign (tip +z), so roll_meas is directly comparable to roll_cmd.
        self.roll_put -= float(self.con.q[self.con.ROLL_JOINT] - q7_pre)
        self.droll_req = 0.0

        from isaacsim.core.utils.types import ArticulationAction
        full = self.art.get_joint_positions()
        tgt = full.copy()
        tgt[self.idx] = self.q
        if self.jidx:
            tgt[self.jidx] = self.grip_half
            # hard kinematic set on top of the drive so the jaws follow the
            # command through tissue. MUST precede apply_action: it rewrites
            # every dof's position target from the measured state.
            self.art.set_joint_positions(np.full(2, self.grip_half),
                                         joint_indices=self.jidx)
        if qd_ff is None:
            self.ctrl.apply_action(ArticulationAction(joint_positions=tgt))
        else:
            vfull = np.zeros_like(tgt)
            vfull[self.idx] = qd_ff
            self.ctrl.apply_action(ArticulationAction(joint_positions=tgt,
                                                      joint_velocities=vfull))

    def _step_gripper(self, grip_frac, dt):
        """Continuous 0..1 opening command, rate-limited, with the same
        force-limited close the teleop uses (jaws stop at the angle where the
        model grasping force reaches `grasp_force`, so tissue is pinched, not
        crushed to a sliver)."""
        env = self.env
        self.tissue_near = False
        self.floor_half = 0.0
        self.probe_c = None
        self.probe_thick = 0.0
        self.grip_cmd_frac = float(np.clip(grip_frac, 0.0, 1.0))
        if not self.jidx:
            return
        want = self.grip_cmd_frac * self.grip_half_max
        # The mouth is probed on BOTH arms and at ANY opening. On the grasper
        # it drives the force floor and the latch, as before. On the cutter it
        # drives nothing -- the blades neither stop nor hold -- but it is what
        # answers "is there appendix between the blades", which the resection
        # metric has to know while the mouth is still OPEN, i.e. before the
        # close that would be the cut. Scoring a cut on tip PROXIMITY instead
        # would pass a tool that closed beside the tissue.
        if env.grasp_on and not self.grasped:
            from laparo_ik import JAW_PIN_AXIS_F
            v_live = env.live_appendix_points()
            near, thick, self.probe_c = tissue_probe(
                v_live, self.p_tip_m, self.a, self.R_fl, self.grip_half,
                JAW_PIN_AXIS_F)
            if near:
                self.tissue_near = True
                self.probe_thick = float(thick)
                if self.grasp_thick is None:
                    self.grasp_thick = float(np.clip(thick, 0.004, 0.011))
            # `want <= grip_half`, i.e. "the command is asking to close at
            # least this far", NOT "the command is exactly zero". The zero test
            # was safe while the only caller was the teleop's binary button,
            # where closing always means want == 0. A RAMPED command breaks it:
            # the jaws follow the ramp down past the force floor before the
            # command reaches zero, so the floor never stops them, they end up
            # shut, and `_step_grasp` -- which needs them resting AT the floor
            # -- can never latch. Measured: with a 2.5 s close ramp, every
            # stage failed and nothing was ever grasped.
            if (self.grips and self.tissue_near
                    and self.grasp_thick is not None
                    and want <= self.grip_half + 1e-9):
                k_strain = 0.075
                strain_stop = float(np.clip(
                    env.cfg["grasp_force"] / k_strain, 0.05, 0.85))
                gap = self.grasp_thick * (1.0 - strain_stop)
                self.floor_half = math.asin(
                    min(1.0, (gap / 2.0) / env.cfg["jaw_length"]))
        # forget the captured tissue thickness only when the jaws are being
        # OPENED (relative to where they are), not on any non-zero command --
        # otherwise a ramp resets it on every tick of the close
        if want > self.grip_half + 1e-9 or \
                (not self.grasped and not self.tissue_near):
            self.grasp_thick = None
        step = math.radians(env.cfg["gripper_rate_dps"]) / 2.0 * dt
        prev = self.grip_half
        if self.grasped and want <= 1e-6:
            self.grip_half = float(np.clip(self.grip_hold,
                                           self.grip_half - step,
                                           self.grip_half + step))
        else:
            self.grip_half = float(np.clip(want, self.grip_half - step,
                                           self.grip_half + step))
            if (self.tissue_near and self.floor_half > 0.0
                    and prev >= self.floor_half):
                self.grip_half = max(self.grip_half, self.floor_half)
        self._step_blade_phase(abs(self.grip_half - prev) > 1e-6, dt)
        self._step_cut(prev)

    def _step_cut(self, prev_half):
        """Watch the blades for a CUT: a close that goes all the way through
        tissue that was in the mouth when it started.

        Nothing is severed -- PhysX cannot re-mesh a cooked deformable, so the
        appendix stays whole and this only RECORDS the gesture. But it records
        it strictly, which is the point: the blades must be open past
        `cut_open_frac` with appendix actually between them (tissue_probe, the
        measured blade geometry -- not tip proximity), and must then close past
        `cut_shut_frac` without the tissue having left the mouth. Closing on
        nothing, closing beside the tissue, or nudging the tissue out on the
        way down all fail to arm or break the arm.

        `cut_site` is where it happened -- the tissue centroid in the mouth,
        which the scorer needs to tell a cut between the grasp and the base
        from a cut anywhere else on the appendix.
        """
        if self.grips or not self.jidx:
            return
        env = self.env
        frac = self.grip_half / max(self.grip_half_max, 1e-9)
        prev_frac = prev_half / max(self.grip_half_max, 1e-9)
        if self.tissue_near and frac >= env.cfg["cut_open_frac"]:
            # open, with tissue between the blades: a cut may now begin
            self.cut_armed = True
            self.cut_site = (None if self.probe_c is None
                             else np.asarray(self.probe_c, np.float64).copy())
        elif self.cut_armed and not self.tissue_near and frac > prev_frac:
            # tissue left the mouth while OPENING -- the setup is stale
            self.cut_armed = False
        if (self.cut_armed and frac <= env.cfg["cut_shut_frac"]
                and prev_frac > env.cfg["cut_shut_frac"]):
            if self.tissue_near and self.probe_c is not None:
                self.cut_site = np.asarray(self.probe_c, np.float64).copy()
            self.cut_armed = False
            if self.cut_site is not None:
                self.n_cuts += 1
                self.cut_done = True
                self.cut_t = env.sim_t
                _log(f"[{self.name:>5}] CUT #{self.n_cuts} at "
                     f"{np.round(self.cut_site, 4).tolist()} "
                     f"(blades {prev_frac*100:.0f}% -> {frac*100:.0f}% open "
                     f"through tissue; the appendix is NOT severed -- the "
                     f"gesture is recorded, not simulated)")

    def _step_grasp(self, grip_frac):
        """Latch/release the appendix grip handles, exactly as the teleop."""
        from pxr import Gf
        env = self.env
        if not (env.grasp_on and self.jidx and self.grips):
            return
        if not self.grasped:
            settled = (self.tissue_near and self.grip_cmd_frac <= 1e-6
                       and self.floor_half > 0.0
                       and abs(self.grip_half - self.floor_half)
                       <= math.radians(1.5) and self.probe_c is not None)
            if settled:
                dom, dom_d = env.dominant_organ(self.p_tip_m)
                if dom != "appendix":
                    v_ap = env.live_appendix_points()
                    ap_d = (float(np.linalg.norm(v_ap - self.p_tip_m,
                                                 axis=1).min())
                            if v_ap is not None else 1e9)
                    if dom_d < ap_d - 0.003:
                        settled = False
            if settled:
                free = [c for c in env.clusters if c["owner"] is None]
                if free:
                    gcp = {id(c): env.cluster_pos(c) for c in free}
                    gc = min(free, key=lambda c: float(
                        np.linalg.norm(gcp[id(c)] - self.probe_c)))
                    d_gc = float(np.linalg.norm(gcp[id(gc)] - self.probe_c))
                    if d_gc < env.cfg["grasp_radius"]:
                        self.grasp_cluster = gc
                        gc["owner"] = self.name
                        # LATCH IN PLACE. The handle must be moved onto the
                        # tissue BEFORE the attachment is enabled.
                        #
                        # The attachment welds its sim vertices at offsets
                        # stored relative to the handle's origin, so enabling
                        # it teleports the tissue to wherever the handle is.
                        # The handle sits at its AUTHORED rest, and by the time
                        # a policy has approached and closed, the organ has
                        # usually been pushed somewhere else -- 22 mm, on the
                        # scripted baseline. Enabling then snapped 22 mm of
                        # tissue in one tick, and the FEM threw the instrument
                        # hard enough to put the shaft 13 mm off a 6 mm port.
                        # Seeding the handle with the tissue's LIVE position
                        # makes the constraint satisfied at the instant it is
                        # switched on, so there is nothing to snap.
                        here = gcp[id(gc)]
                        gc["op"].Set(Gf.Vec3d(*[float(x) for x in here]))
                        gc["en"].Set(True)
                        self.follow_h0 = np.asarray(here, np.float64).copy()
                        self.follow_t0 = self.p_tip_m.copy()
                        self.grasped = True
                        self.grip_hold = max(self.floor_half,
                                             math.radians(4.0))
        else:
            if self.grip_cmd_frac > 0.05 and self.grip_half > math.radians(8.0):
                self.grasp_cluster["en"].Set(False)
                self.grasp_cluster["op"].Set(Gf.Vec3d(
                    *[float(x) for x in self.grasp_cluster["rest"]]))
                self.grasp_cluster["owner"] = None
                self.grasp_cluster = None
                self.grasped = False
                self.grip_hold = 0.0
            else:
                h = self.follow_h0 + (self.p_tip_m - self.follow_t0)
                self.grasp_cluster["op"].Set(
                    Gf.Vec3d(*[float(x) for x in h]))

    def post_tick(self):
        import trocar
        self.refresh_measured()
        self.lag = self.p_tip_m - self.p_shaped
        p_c = self.p_tip_m + self.a * np.dot(self.p_rcm - self.p_tip_m, self.a)
        self.rcm_err = float(np.linalg.norm(self.p_rcm - p_c))
        self.bite = float(trocar.port_geometry(self.p_rcm, self.u_in,
                                               self.p_tip_m, self.a)["bite"][0])
        self.sample_wrench()


# ==========================================================================
# the environment
# ==========================================================================
class LaparoVLAEnv:
    """The surgical cell as a step()/reset() environment."""

    def __init__(self, headless=True, **kw):
        unknown = set(kw) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown option(s): {sorted(unknown)}")
        self.cfg = dict(DEFAULTS)
        self.cfg.update(kw)
        self.headless = bool(headless)
        self.app = _boot(self.headless)

        import omni.timeline
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
        self._omni_usd = omni.usd
        self._timeline = omni.timeline.get_timeline_interface()

        scene_path = ROOT / self.cfg["scene"]
        if not scene_path.exists():
            raise FileNotFoundError(
                f"{scene_path} not found -- build it first:\n"
                f"  ~/isaacsim-5.1/python.sh isaac/build_surgical_cell.py")
        omni.usd.get_context().open_stage(str(scene_path))
        self.stage = omni.usd.get_context().get_stage()
        _log(f"opened {scene_path}")

        # THE CONTROLLER, with one deliberate change from the shipped BEST.
        #
        # `manip_hz`: this env runs the control law on EVERY physics substep
        # (480 Hz), not at control_hz, so the controller alone would cost two
        # arms x 480 Hz x 5.8 ms = 5.6 s of CPU per simulated second. 64 of
        # the 88 forward-kinematics calls in a tick are the manipulability
        # climb's nested finite differences, and that term steers the ELBOW,
        # which moves in fractions of a degree per second. Holding it at 60 Hz
        # takes the tick to 1.5 ms. TELEOP_RESPONSIVENESS.md section 2 has the
        # A/B: worst RCM error, bite, tracking lag and shaft-roll error are
        # identical to the last digit, and holding it 16 ms is inside the
        # term's own nested-finite-difference noise floor.
        #
        # `laparo_control.BEST` is left at manip_hz=0 on purpose so that it
        # cannot change any caller's behaviour silently -- so it is opted into
        # HERE, and `manip_hz=0` restores bit-exact BEST.
        from laparo_control import BEST
        self.control_cfg = (
            BEST.replace(manip_hz=self.cfg["manip_hz"],
                         name=f"{BEST.name}+manip{self.cfg['manip_hz']:g}")
            if self.cfg["manip_hz"] else BEST)
        self.k_avoid, self.avoid_dist, self.avoid_radius = 6.0, 0.08, 0.05

        # the phantom dome is a visual/registration aid, not a surface the
        # instruments work against; teleop drops its collision too
        prim = self.stage.GetPrimAtPath("/World/colliders/phantom")
        if prim:
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            for d in Usd.PrimRange(prim):
                if d.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(d).GetCollisionEnabledAttr().Set(False)

        tp = self.stage.GetPrimAtPath("/World/trocarPorts")
        self.R_ports = _world_transform(tp)[:3, :3] if tp else np.eye(3)
        self.start_tips = self._aim_start_tips()

        from start_pose import MIRROR
        self.rigs = {}
        prev = None
        for nm in ARMS:
            rig = ArmRig(nm, self, seed_q=None if prev is None else prev * MIRROR)
            self.rigs[nm] = rig
            prev = rig.q0
        self.left, self.right = self.rigs["left"], self.rigs["right"]

        # rig frame: origin at the midpoint of the two FR3 bases, +x from left
        # base to right base, +z world up
        bl = self.left.T_base[:3, 3]
        br = self.right.T_base[:3, 3]
        origin = 0.5 * (bl + br)
        x = br - bl
        x[2] = 0.0
        x = (x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9
             else np.array([1.0, 0.0, 0.0]))
        z = np.array([0.0, 0.0, 1.0])
        y = np.cross(z, x)
        self.T_world_rig = _T(np.column_stack([x, y, z]), origin)
        self.T_rig_world = _inv_T(self.T_world_rig)
        _log(f"rig frame origin (world) {np.round(origin, 4).tolist()}, "
             f"base separation {np.linalg.norm(br - bl)*1000:.0f} mm")

        # physics
        from isaacsim.core.api import SimulationContext
        self._assert_physics()
        self.sim = SimulationContext(stage_units_in_meters=1.0)
        self._assert_physics()
        self.sim.initialize_physics()
        self._assert_physics()
        for rig in self.rigs.values():
            rig.attach()
        _log(f"controller: {self.control_cfg.label()}")

        self._timeline.play()
        for _ in range(60):
            self.app.update()
        _log("pumped 60 app updates for async deformable cooking")
        self._report_organs()
        self._arm_grasp_assist()

        self.cams = {}
        self._build_cameras()

        self.dt_phys = 1.0 / float(self.cfg["physics_hz"])
        self.n_sub = max(1, int(round(self.cfg["physics_hz"]
                                      / self.cfg["control_hz"])))
        self.dt_ctrl = self.n_sub * self.dt_phys
        _log(f"control {1.0/self.dt_ctrl:.1f} Hz = {self.n_sub} physics ticks "
             f"of {self.dt_phys*1000:.2f} ms; chunk {self.cfg['chunk']} "
             f"({self.cfg['chunk']*self.dt_ctrl*1000:.0f} ms per step())")

        self._rt = {"stage": None, "frame": -1, "pts": {}}
        self.frame = 0
        self.step_count = 0
        self.sim_t = 0.0
        self._closed = False
        self._reset_done = False
        self._rcm_bad_run = 0
        self.n_rcm_violation = 0

    # ---- boot helpers ----------------------------------------------------
    def _assert_physics(self):
        """GPU dynamics + the deformable-safe timestep, re-asserted.
        SimulationContext's boot actively authors CPU/60 Hz over whatever the
        cell carries, which kills every organ."""
        from pxr import PhysxSchema, UsdPhysics
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdPhysics.Scene):
                continue
            px = PhysxSchema.PhysxSceneAPI.Apply(prim)
            px.CreateEnableGPUDynamicsAttr().Set(True)
            px.CreateBroadphaseTypeAttr().Set("GPU")
            px.CreateSolverTypeAttr().Set("TGS")
            px.CreateTimeStepsPerSecondAttr().Set(self.cfg["physics_hz"])
            px.CreateGpuCollisionStackSizeAttr().Set(2 * 67108864)
            px.CreateGpuMaxSoftBodyContactsAttr().Set(2 * 1048576)

    def _aim_start_tips(self):
        from laparo_ik import port_inward_normal
        from start_pose import endoscope_camera, solve_start_tips, spawn_tips
        cam = None
        if self.cfg["start_aim"] != "approach":
            try:
                cam = endoscope_camera(self.stage)
            except Exception as exc:
                _log(f"start aim unavailable ({exc}) -- using the port normals")
                return {}
        ports, arm_of_port = {}, {}
        for nm, cfg in ARM_CFG.items():
            for i in (2, 3):
                prim = self.stage.GetPrimAtPath(f"/World/trocarPorts/port_{i:02d}")
                if prim and _world_transform(prim)[0, 3] * cfg["port_sign"] > 0:
                    ports[i] = (_world_transform(prim)[:3, 3],
                                self.R_ports @ port_inward_normal(i))
                    arm_of_port[i] = nm
        if len(ports) != 2:
            return {}
        organ_pts = []
        org = self.stage.GetPrimAtPath("/World/organs")
        if org and org.IsValid():
            for kid in org.GetChildren():
                rp = kid.GetChild("render")
                pa = rp.GetAttribute("points").Get() if rp else None
                if not pa:
                    continue
                M = _world_transform(rp)
                v = np.asarray([[p[0], p[1], p[2]] for p in pa], np.float64)
                organ_pts.append((M[:3, :3] @ v.T).T + M[:3, 3])
        organ_pts = np.vstack(organ_pts) if organ_pts else None
        try:
            if self.cfg["start_aim"] == "approach":
                over = {nm: self.cfg.get(f"approach_tip_{nm}", "")
                        for nm in ARMS}
                sol = spawn_tips(ports, arm_of_port, self.cfg["insert"], over)
            else:
                sol = solve_start_tips(cam, ports, insert=self.cfg["insert"],
                                       min_sep=0.045,
                                       max_tilt_deg=self.cfg["start_max_tilt"],
                                       v_frac=0.5, organ_pts=organ_pts,
                                       min_clear=0.008)
                sol.pop("_meta", None)
            return {arm_of_port[k]: v["tip"] for k, v in sol.items()}
        except Exception as exc:
            _log(f"start aim FAILED ({exc}) -- using the port normals")
            return {}

    def _report_organs(self):
        self.organ_names = []
        org = self.stage.GetPrimAtPath("/World/organs")
        if not (org and org.IsValid()):
            return
        for kid in org.GetChildren():
            sim_m = kid.GetChild("sim")
            pts = sim_m.GetAttribute("points").Get() if sim_m else None
            rp = kid.GetChild("render")
            if rp and rp.GetAttribute("points").Get():
                self.organ_names.append(kid.GetName())
            _log(f"organ {kid.GetName():9s}: "
                 f"{f'cooked, {len(pts)} sim nodes' if pts else 'NOT cooked'}")

    def _arm_grasp_assist(self):
        from pxr import UsdGeom
        self.clusters = []
        self.grasp_on = False
        if not self.cfg["grasp_assist"]:
            _log("grasp assist DISABLED by configuration")
            return
        pins = self.stage.GetPrimAtPath("/World/organPins")
        appx = self.stage.GetPrimAtPath("/World/organs/appendix/render")
        if not (pins and pins.IsValid() and appx and appx.IsValid()):
            _log("no appendix grip clusters in this scene -- grasp assist off")
            return
        for hp in sorted(pins.GetChildren(), key=lambda p: p.GetName()):
            nm = hp.GetName()
            if not (nm == "appendixGrip" or nm.startswith("appendixGrip_")):
                continue
            att_name = ("grip_appendix" if nm == "appendixGrip"
                        else "grip_appendix_" + nm.split("_", 1)[1])
            att = pins.GetChild(att_name)
            if not (att and att.IsValid()):
                continue
            hxf = UsdGeom.Xformable(hp)
            top = next((op for op in hxf.GetOrderedXformOps()
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
                       None) or hxf.AddTranslateOp()
            hpv = top.Get()
            ia = att.GetAttribute("omniphysics:vtxIndicesSrc0")
            iv = ia.Get() if (ia and ia.IsValid()) else None
            self.clusters.append(dict(
                name=nm, op=top,
                en=att.GetAttribute("omniphysics:attachmentEnabled"),
                rest=np.array([hpv[0], hpv[1], hpv[2]]),
                idx=(np.asarray(iv, dtype=np.int64)
                     if iv is not None and len(iv) else None),
                owner=None))
        self.grasp_on = len(self.clusters) > 0
        for c in self.clusters:
            c["en"].Set(False)
        # The appendix's FAR end is pinned to the world (build_surgical_cell
        # author_pins: appendectomy setup -- the end away from the left robot
        # is a fixed anchor). A grasp taken next to that pin is a pull against
        # a rigid world constraint, which is what threw the tool hard enough at
        # load onset to saturate the command leash and put the shaft 16 mm off
        # the port. Locate it so the distance to it can be reported, and so a
        # policy (or a scorer) can tell the difference between "grabbed badly"
        # and "the controller failed".
        self.pin_idx = None
        pin = pins.GetChild("pin_appendix") if pins else None
        if pin and pin.IsValid():
            ia = pin.GetAttribute("omniphysics:vtxIndicesSrc0")
            iv = ia.Get() if (ia and ia.IsValid()) else None
            if iv is not None and len(iv):
                self.pin_idx = np.asarray(iv, dtype=np.int64)
        _log(f"grasp assist: {len(self.clusters)} appendix grip cluster(s), "
             f"all released at rest; world pin "
             + (f"= {len(self.pin_idx)} welded sim nodes"
                if self.pin_idx is not None else "NOT found"))

    def _res_of(self, path, ns, default):
        """The resolution the cell AUTHORED for this camera. The intrinsics
        were calibrated at it, so reading it beats hard-coding it."""
        prim = self.stage.GetPrimAtPath(path)
        if not (prim and prim.IsValid()):
            return default
        w = prim.GetAttribute(f"{ns}:width").Get()
        h = prim.GetAttribute(f"{ns}:height").Get()
        return (int(w), int(h)) if w and h else default

    def _build_cameras(self):
        ew, eh = self._res_of(ENDO_CAM, "endocam", (1280, 720))
        dw, dh = self._res_of(D405_CAM, "d405", (848, 480))
        self.cams["endoscope"] = CamSensor(
            self.stage, ENDO_CAM, "endoscope", ew, eh,
            want_depth=bool(self.cfg["endo_depth"]),
            vignette=float(self.cfg["endo_vignette"]),
            vignette_soft=float(self.cfg["endo_vignette_soft"]))
        self.cams["d405"] = CamSensor(self.stage, D405_CAM, "d405", dw, dh,
                                      want_depth=True)
        if self.cfg["stereo"]:
            self._add_stereo_partner()
        for nm, c in self.cams.items():
            _log(f"camera {nm:12s} {c.W}x{c.H}  fx={c.K[0,0]:.1f} "
                 f"fy={c.K[1,1]:.1f} cx={c.K[0,2]:.1f} cy={c.K[1,2]:.1f}  "
                 f"FOV {c.hfov:.1f}x{c.vfov:.1f} deg"
                 f"{'  +depth' if c.want_depth else ''}")

    def _add_stereo_partner(self):
        """The D405 is a stereo module (18 mm baseline): give it its second
        imager, so a policy can run its own matcher instead of taking the
        renderer's ground-truth z.

        Authored as a CHILD of the calibrated camera with a single local
        translate along +x (image-right). Parenting is what makes the offset a
        genuine baseline: the child inherits the parent's orientation exactly,
        so the pair is rectified by construction and the existing, view-matched
        camera stays untouched as the LEFT imager."""
        from pxr import Gf, Sdf, UsdGeom
        src = self.stage.GetPrimAtPath(D405_CAM)
        if not (src and src.IsValid()):
            _log("no D405 camera in this scene -- stereo partner skipped")
            return
        path = D405_CAM + "/d405_right"
        dst = self.stage.GetPrimAtPath(path)
        if not (dst and dst.IsValid()):
            dc = UsdGeom.Camera.Define(self.stage, path)
            dst = dc.GetPrim()
            sc = UsdGeom.Camera(src)
            dc.CreateFocalLengthAttr().Set(sc.GetFocalLengthAttr().Get())
            dc.CreateHorizontalApertureAttr().Set(
                sc.GetHorizontalApertureAttr().Get())
            dc.CreateVerticalApertureAttr().Set(
                sc.GetVerticalApertureAttr().Get())
            dc.CreateHorizontalApertureOffsetAttr().Set(
                float(sc.GetHorizontalApertureOffsetAttr().Get() or 0.0))
            dc.CreateVerticalApertureOffsetAttr().Set(
                float(sc.GetVerticalApertureOffsetAttr().Get() or 0.0))
            dc.CreateClippingRangeAttr().Set(sc.GetClippingRangeAttr().Get())
            UsdGeom.Xformable(dst).AddTranslateOp().Set(
                Gf.Vec3d(float(self.cfg["d405_baseline"]), 0.0, 0.0))
            dst.CreateAttribute("d405:width", Sdf.ValueTypeNames.Int).Set(848)
            dst.CreateAttribute("d405:height", Sdf.ValueTypeNames.Int).Set(480)
        self.cams["d405_right"] = CamSensor(self.stage, path, "d405_right",
                                            848, 480, want_depth=False)
        _log(f"D405 stereo partner at +{self.cfg['d405_baseline']*1000:.0f} mm "
             f"baseline -> {path}")

    # ---- live tissue -----------------------------------------------------
    def live_points(self, path):
        if self._rt["frame"] != self.frame:
            self._rt["frame"], self._rt["pts"] = self.frame, {}
        if path in self._rt["pts"]:
            return self._rt["pts"][path]
        if self._rt["stage"] is None:
            import usdrt
            self._rt["stage"] = usdrt.Usd.Stage.Attach(
                self._omni_usd.get_context().get_stage_id())
        p = self._rt["stage"].GetPrimAtPath(path)
        a = p.GetAttribute("points") if p else None
        val = a.Get() if a and a.IsValid() else None
        pts = (np.array([[v[0], v[1], v[2]] for v in val])
               if val is not None and len(val) else None)
        self._rt["pts"][path] = pts
        return pts

    def live_appendix_points(self):
        return self.live_points("/World/organs/appendix/render")

    def cluster_pos(self, c):
        if c["idx"] is None:
            return c["rest"]
        v = self.live_points("/World/organs/appendix/sim")
        if v is None or len(v) <= c["idx"].max():
            return c["rest"]
        return v[c["idx"]].mean(axis=0)

    def dominant_organ(self, p_tip):
        best, bestd = None, 1e9
        for nm in self.organ_names:
            pts = self.live_points(f"/World/organs/{nm}/render")
            if pts is None:
                continue
            d = float(np.linalg.norm(pts - p_tip, axis=1).min())
            if d < bestd:
                best, bestd = nm, d
        return best, bestd

    # ---- gym-ish surface -------------------------------------------------
    def _settle(self, seconds, grip=0.0):
        """Hold the current command for `seconds` of simulated time."""
        for _ in range(int(seconds * self.cfg["physics_hz"])):
            for rig in self.rigs.values():
                rig.apply(np.zeros(3), np.zeros(3), grip, self.dt_phys,
                          np.eye(3))
            self.sim.step(render=False)
            for rig in self.rigs.values():
                rig.post_tick()
            self.frame += 1

    def _identify_forces(self):
        """Free-space excitation -> the payload/bias model behind the force
        channel. Done once per process (the model is a property of the arm and
        its instrument, not of the tissue) and cached to
        `out/vla_force_calib.json` so later runs start instantly."""
        cache = ROOT / "out" / "vla_force_calib.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                if all(nm in data for nm in ARMS):
                    for nm in ARMS:
                        self.rigs[nm].wrench_model = data[nm]
                    _log(f"force model loaded from {cache} (delete it to "
                         f"re-identify)")
                    return
            except Exception as exc:
                _log(f"could not read {cache} ({exc}) -- re-identifying")

        # MOVE-DWELL-SAMPLE, not a continuous sweep. The model is static, so a
        # sample taken while the arm is moving is a measurement of the
        # excitation, not of the payload: a first attempt that swept
        # continuously and gated on joint speed threw away every sample it
        # took. Here the arm is driven to a pose, allowed to come to rest, and
        # only then read.
        hz = self.cfg["physics_hz"]
        MOVE, DWELL, READ = 0.45, 0.35, 0.12         # seconds per waypoint
        LIFT = 0.025          # m of world +z clearance held for the whole run

        # (roll deg, world offset mm, gripper %) relative to the start pose.
        # Roll separates the payload from the constant bias; the translations
        # make the COM observable; the gripper sweep gives the jaw baseline.
        WPTS = []
        for roll in (0.0, 18.0, 35.0, 18.0, 0.0, -18.0, -35.0, -18.0, 0.0):
            for off in ((0, 0, 0), (12, 0, 0), (0, 12, 0), (0, 0, 10),
                        (-12, 0, 0), (0, -12, 0)):
                WPTS.append((roll, np.array(off, float) * 1e-3))
        _log(f"identifying the force model: {len(WPTS)} static poses x "
             f"{MOVE+DWELL:.1f} s = "
             f"{len(WPTS)*(MOVE+DWELL)+2.0:.0f} s of simulated time "
             f"(one time, then cached)")

        base = {nm: self.rigs[nm].p_cmd.copy() for nm in ARMS}
        roll_now = {nm: 0.0 for nm in ARMS}

        def drive(seconds, target_off, droll_total, grip, sample_after=None,
                  what=None):
            n = max(1, int(seconds * hz))
            for i in range(n):
                R_step = {nm: np.column_stack(self.rigs[nm].tool_axes())
                          for nm in ARMS}
                for nm, rig in self.rigs.items():
                    want = base[nm] + target_off
                    d_w = (want - rig.p_cmd) / max(n - i, 1)
                    rig.apply(R_step[nm].T @ d_w,
                              np.array([0.0, 0.0, droll_total / n]), grip,
                              self.dt_phys, R_step[nm])
                self.sim.step(render=False)
                for rig in self.rigs.values():
                    rig.post_tick()
                self.frame += 1
                if sample_after is not None and i >= sample_after * n:
                    for rig in self.rigs.values():
                        rig.calib_sample(**(what or {}))

        # hold clear of the organ bed for the whole identification: the start
        # pose only guarantees 8 mm of clearance and the offsets above are
        # wider than that. Contact here would be fitted as payload and then
        # subtracted out of every later reading.
        up = np.array([0.0, 0.0, LIFT])
        drive(1.5, up, 0.0, 0.0)
        # A) the payload, jaws SHUT throughout
        for roll_deg, off in WPTS:
            d_roll = math.radians(roll_deg) - roll_now[ARMS[0]]
            for nm in ARMS:
                roll_now[nm] = math.radians(roll_deg)
            drive(MOVE, up + off, d_roll, 0.0)
            drive(DWELL, up + off, 0.0, 0.0, sample_after=1.0 - READ / DWELL,
                  what=dict(wrench=True))
        # B) the jaw baseline, arm held still, gripper stepped through its range
        for g in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
            drive(0.5, up, 0.0, g)
            drive(0.3, up, 0.0, g, sample_after=0.5,
                  what=dict(wrench=False, jaw=True))
        for rig in self.rigs.values():
            rig.fit_wrench_model()
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(
                {nm: self.rigs[nm].wrench_model for nm in ARMS}, indent=2))
            _log(f"force model cached -> {cache}")
        except Exception as exc:
            _log(f"could not cache the force model ({exc})")

    def reset(self, seed=None):
        """Return the instruments to the start pose and settle the tissue.

        First call also identifies the force model, which needs the arms to
        move in free space -- so it happens BEFORE the pose is restored, and
        the episode always begins from the same, settled start pose."""
        if seed is not None:
            np.random.seed(int(seed))
        for c in self.clusters:
            c["en"].Set(False)
            c["owner"] = None
        for rig in self.rigs.values():
            rig.reset_state()
        # the tissue baseline is per EPISODE: keeping it would make episode 2's
        # "lift" the displacement since episode 1 started
        for k in ("_appendix_c0", "_appendix_z0"):
            if hasattr(self, k):
                delattr(self, k)
        self._settle(0.5)
        if any(r.wrench_model is None for r in self.rigs.values()):
            self._identify_forces()
            for rig in self.rigs.values():        # back to the start pose
                rig.reset_state()
            self._settle(0.5)
        # Fly both tips from the spawn to the working pose BEFORE anything the
        # episode is measured against: the tare below zeroes the force channels
        # at whatever pose the arms are in, and the first observation is
        # returned from here, so the policy's step 0 is the arms on station.
        self._approach()
        self._tare_forces()
        self._refresh_obstacles()
        self.frame += 1
        self.step_count = 0
        self._rcm_bad_run = 0
        self.n_rcm_violation = 0
        self.sim_t = 0.0
        self._reset_done = True
        # pump until the annotators publish, so the first observation of an
        # episode is a real frame rather than a held zero
        for _ in range(4):
            self._render_once()
            if all(not c.read().get("stale", False) for c in self.cams.values()):
                break
        return self.observe()

    def _approach(self):
        """Drive both tips from the spawn to the measured working pose.

        The instruments SPAWN 10 cm past their ports, short of the organ bed --
        a deeper spawn has the start solver placing tips inside tissue (the
        depth is a hard constraint, so its clearance term can only choose the
        least-buried direction) and the episode opens with the organs already
        deformed around a tool and pushed out of the overlap.

        Getting from there to the surgical pose is not the policy's job, and
        it is not something a policy should have to spend its first hundred
        actions on either. It is a fixed, scripted move, run here under the
        ordinary control law -- so the trocar constraint, the depth floor, the
        tilt cone and the leash all hold during the insertion exactly as they
        do afterwards, and nothing about the arm's state at handover is
        special. `rig.apply` with zero deltas leaves p_cmd where this puts it
        and runs the whole clamp chain over it.
        """
        from start_pose import approach_command, approach_duration, approach_tip
        if not self.cfg["approach"]:
            _log("approach disabled -- the arms stay 10 cm past their ports")
            return
        plan = {}
        for nm, rig in self.rigs.items():
            try:
                p_to = approach_tip(nm, self.cfg.get(f"approach_tip_{nm}", ""))
            except (KeyError, ValueError) as exc:
                _log(f"{nm}: no approach target ({exc}) -- staying at the spawn")
                continue
            p_from = rig.p_cmd.copy()
            plan[nm] = dict(
                p_from=p_from, p_to=p_to,
                T=approach_duration(rig.p_rcm, p_from, p_to,
                                    self.cfg["approach_speed"]))
        if not plan:
            return
        T = max(p["T"] for p in plan.values())
        n = int(math.ceil((T + self.cfg["approach_timeout_s"])
                          * self.cfg["physics_hz"]))
        legs = ", ".join(
            "%s %.0f mm" % (nm, np.linalg.norm(pl["p_to"] - pl["p_from"]) * 1e3)
            for nm, pl in sorted(plan.items()))
        _log(f"approach: {legs} over {T:.1f} s at "
             f"{self.cfg['approach_speed']*100:.0f} cm/s")
        for i in range(n):
            s = min(1.0, (i + 1) / max(T * self.cfg["physics_hz"], 1.0))
            for nm, pl in plan.items():
                rig = self.rigs[nm]
                rig.p_cmd = approach_command(rig.p_rcm, pl["p_from"],
                                             pl["p_to"], s)
                rig.apply(np.zeros(3), np.zeros(3), 0.0, self.dt_phys,
                          np.eye(3))
            self.sim.step(render=False)
            for rig in self.rigs.values():
                rig.post_tick()
            self.frame += 1
            if s >= 1.0 and all(
                    np.linalg.norm(self.rigs[nm].p_tip_m - pl["p_to"])
                    <= self.cfg["approach_tol_m"] for nm, pl in plan.items()):
                break
        for nm, pl in sorted(plan.items()):
            err = float(np.linalg.norm(self.rigs[nm].p_tip_m - pl["p_to"]))
            _log(f"  {nm}: tip {np.round(self.rigs[nm].p_tip_m, 4).tolist()}, "
                 f"{err*1000:.1f} mm from target"
                 + ("" if err <= self.cfg["approach_tol_m"]
                    else f" (over the {self.cfg['approach_tol_m']*1000:.0f} mm "
                         f"tolerance -- handing over from here anyway)"))
        self._settle(0.25)

    def _tare_forces(self):
        """Zero every force channel at the settled start pose."""
        for rig in self.rigs.values():
            rig.w_tare = np.zeros(6)
        acc = {nm: [] for nm in ARMS}
        for _ in range(int(0.25 * self.cfg["physics_hz"])):
            for rig in self.rigs.values():
                rig.apply(np.zeros(3), np.zeros(3), 0.0, self.dt_phys,
                          np.eye(3))
            self.sim.step(render=False)
            for nm, rig in self.rigs.items():
                rig.post_tick()
                w = rig.wrench_tool_now()
                if w is not None:
                    acc[nm].append(w)
            self.frame += 1
        for nm, rig in self.rigs.items():
            rig.tare(acc[nm])
        for rig in self.rigs.values():          # drop the pre-tare accumulator
            rig._wacc, rig._jacc, rig._wn = np.zeros(6), 0.0, 0

    def step(self, action):
        """Apply one action chunk. Returns (obs, info)."""
        if not self._reset_done:
            raise RuntimeError("call reset() before step()")
        a = Action.check(action, self.cfg["chunk"])
        K = a.shape[1]
        stats = {nm: dict(rot_rejected=np.zeros(3), depth_clamped=0,
                          cone_clamped=0) for nm in ARMS}
        for k in range(K):
            R_step = {nm: np.column_stack(self.rigs[nm].tool_axes())
                      for nm in ARMS}
            for nm in ARMS:
                stats[nm]["rot_rejected"] += a[ARMS.index(nm), k, 3:6] * \
                    np.array([1.0, 1.0, 0.0])
            for _ in range(self.n_sub):
                for i, nm in enumerate(ARMS):
                    rig = self.rigs[nm]
                    rig.apply(a[i, k, 0:3] / self.n_sub,
                              a[i, k, 3:6] / self.n_sub,
                              a[i, k, 6] / 100.0,
                              self.dt_phys, R_step[nm])
                self.sim.step(render=False)
                for rig in self.rigs.values():
                    rig.post_tick()
                self.frame += 1
                self.sim_t += self.dt_phys
            self.step_count += 1
        self._refresh_obstacles()
        self._render_once()
        obs = self.observe()
        info = {nm: dict(
            rot_rejected=stats[nm]["rot_rejected"].astype(np.float32),
            depth_clamped=int(self.rigs[nm].n_depth_clamped),
            cone_clamped=int(self.rigs[nm].n_cone_cmd),
            leash_clamped=int(self.rigs[nm].n_leash),
            rcm_err_mm=float(self.rigs[nm].rcm_err * 1000.0),
            trocar_bite_mm=float(self.rigs[nm].bite * 1000.0),
            track_lag_mm=float(np.linalg.norm(self.rigs[nm].lag) * 1000.0),
            cmd_error_mm=float(np.linalg.norm(self.rigs[nm].p_cmd
                                              - self.rigs[nm].p_tip_m) * 1000.0),
            roll_backlog_deg=float(math.degrees(self.rigs[nm].roll_cmd
                                                - self.rigs[nm].roll_put)),
        ) for nm in ARMS}
        info["sub_steps"] = K
        # A run that scores well while the shaft is outside its 6 mm port has
        # not succeeded at anything -- but a grasp taking load throws the tool
        # for a step or two and then recovers, so the abort needs persistence,
        # not just a level. Every excursion is counted either way.
        bad = [nm for nm in ARMS
               if self.rigs[nm].rcm_err * 1000.0 > self.cfg["rcm_abort_mm"]]
        self._rcm_bad_run = (self._rcm_bad_run + 1) if bad else 0
        # The excursion counter runs on its OWN level, not the abort's. The
        # abort says "this run is over"; the excursion count says "the shaft
        # left its 6 mm port", and those stopped being the same question when
        # the abort was raised to 10 mm. Counting on the abort level would
        # have made rcm_excursion_frac read 0.0 for a run that spent its whole
        # length 9 mm out of the trocar.
        self.n_rcm_violation += any(
            self.rigs[nm].rcm_err * 1000.0 > self.cfg["rcm_excursion_mm"]
            for nm in ARMS)
        info["rcm_violation"] = bad
        info["rcm_violation_steps"] = int(self._rcm_bad_run)
        info["rcm_violation_total"] = int(self.n_rcm_violation)
        info["terminated"] = bool(
            self._rcm_bad_run >= self.cfg["rcm_abort_steps"])
        info["truncated"] = bool(self.cfg["max_episode_steps"] and
                                 self.step_count >= self.cfg["max_episode_steps"])
        return obs, info

    def _refresh_obstacles(self):
        """Each arm avoids where the OTHER arm actually is right now."""
        if self.k_avoid <= 0.0:
            return
        for nm in ARMS:
            other = self.rigs["right" if nm == "left" else "left"]
            pts = other.chain.fk_points(other.q, other.T_base)
            self.rigs[nm].obs_segs = list(zip(pts[:-1], pts[1:]))

    def _render_once(self):
        """Render + pump the app, so the camera annotators publish this pose.

        `sim.render()` rather than `sim.step(render=True)`: physics has already
        been advanced by exactly the sub-steps the action asked for, and a
        rendering step would sneak an extra tick in on top of it."""
        self.sim.render()
        self.app.update()

    # ---- observation -----------------------------------------------------
    def observe(self):
        obs = dict(t=float(self.sim_t), step=int(self.step_count),
                   control_hz=float(1.0 / self.dt_ctrl),
                   T_world_rig=self.T_world_rig.astype(np.float32))
        for nm, cam in self.cams.items():
            d = cam.read()
            Tw = cam.pose_world()
            d.update(cam.info())
            d["T_world_cam"] = Tw.astype(np.float32)
            d["T_rig_cam"] = (self.T_rig_world @ Tw).astype(np.float32)
            obs[nm] = d
        if "d405_right" in obs:
            obs["d405"]["rgb_right"] = obs["d405_right"].get("rgb")
            obs["d405"]["baseline_m"] = float(self.cfg["d405_baseline"])
            obs["d405"]["T_cam_right"] = (
                _inv_T(obs["d405"]["T_world_cam"].astype(np.float64))
                @ obs["d405_right"]["T_world_cam"].astype(np.float64)
            ).astype(np.float32)
        for nm in ARMS:
            obs[nm] = self._arm_obs(self.rigs[nm])
        obs["left_force"] = self._force_obs(self.left)
        obs["task"] = self._task_obs()
        return obs

    def _to_rig(self, p):
        return self.T_rig_world[:3, :3] @ np.asarray(p) + self.T_rig_world[:3, 3]

    def _task_obs(self):
        """PRIVILEGED ground truth -- for scoring and for scripted baselines,
        NOT for a policy that has to transfer. Nothing here is observable
        through a real endoscope."""
        v = self.live_appendix_points()
        out = dict(target_organ="appendix")
        if v is None:
            return out
        c = v.mean(axis=0)
        if not hasattr(self, "_appendix_c0"):
            self._appendix_c0 = c.copy()
            self._appendix_z0 = float(np.percentile(v[:, 2], 95))
        out["appendix_centroid_rig"] = self._to_rig(c).astype(np.float32)
        out["appendix_lift_m"] = np.float32(
            float(np.percentile(v[:, 2], 95)) - self._appendix_z0)
        out["appendix_moved_m"] = np.float32(
            float(np.linalg.norm(c - self._appendix_c0)))
        h = np.array([self.cluster_pos(cl) for cl in self.clusters]) \
            if self.clusters else np.zeros((0, 3))
        out["handles_rig"] = np.array([self._to_rig(p) for p in h],
                                      np.float32) if len(h) else h.astype(np.float32)
        out["handle_free"] = np.array([cl["owner"] is None
                                       for cl in self.clusters], bool)
        # the world anchor, and how far each handle is from it. Lifting a
        # handle that sits on the pin is lifting against the world.
        pin_w = None
        if self.pin_idx is not None:
            v_sim = self.live_points("/World/organs/appendix/sim")
            if v_sim is not None and len(v_sim) > self.pin_idx.max():
                pin_w = v_sim[self.pin_idx].mean(axis=0)
        if pin_w is not None:
            out["appendix_pin_rig"] = self._to_rig(pin_w).astype(np.float32)
            if len(h):
                out["handle_to_pin_m"] = np.array(
                    [float(np.linalg.norm(p - pin_w)) for p in h], np.float32)
        for nm in ARMS:
            rig = self.rigs[nm]
            out[f"{nm}_tip_to_appendix_m"] = np.float32(
                float(np.linalg.norm(v - rig.p_tip_m, axis=1).min()))
            # distance to the world-anchored base, not the nearest surface
            # point -- this is what a second tool "approaching the base to
            # resect" needs, as opposed to tip_to_appendix which is satisfied
            # by touching the tissue anywhere
            if pin_w is not None:
                out[f"{nm}_tip_to_pin_m"] = np.float32(
                    float(np.linalg.norm(pin_w - rig.p_tip_m)))
        out["grasped_any"] = bool(any(r.grasped for r in self.rigs.values()))
        self._resection_obs(out, v, pin_w)
        return out

    def _resection_obs(self, out, v, pin_w):
        """The staged appendectomy: grasp the mid-body, lift it, and cut the
        stretch of appendix BETWEEN that grasp and the body.

        Every stage is reported as ground truth here and combined into a
        verdict by vla_rollout.Episode.score -- this side stays descriptive so
        the thresholds can move without re-running anything.

        The appendix is parameterised once, by its own grip clusters: they were
        authored as a disjoint ~1 cm Voronoi split along its long axis from the
        free end (#0) to the pinned base (#K-1), so a cluster INDEX is already
        a position along the organ. `grasp_frac` normalises that to 0..1, and
        the resection segment is everything beyond the held cluster, toward the
        base -- which is exactly "between what the left hand is holding and the
        body".
        """
        K = len(self.clusters)
        out["n_grip_clusters"] = np.int32(K)
        held = next((i for i, c in enumerate(self.clusters)
                     if c["owner"] == "left"), -1)
        out["grasp_index"] = np.int32(held)
        lo, hi = self.cfg["grasp_band"]
        # 0 = free tip, 1 = base. With K clusters the centres sit at
        # (i+0.5)/K, so a single cluster does not read as both ends.
        gfrac = (held + 0.5) / K if (held >= 0 and K) else float("nan")
        out["grasp_frac"] = np.float32(gfrac)
        out["grasp_in_band"] = bool(held >= 0 and lo <= gfrac <= hi)
        if K:
            out["grasp_band"] = np.array([lo, hi], np.float32)

        # the resection segment: appendix vertices beyond the held cluster,
        # on the base side. Measured on the LIVE surface, so it follows the
        # tissue as the left hand lifts it.
        seg = None
        if held >= 0 and v is not None and len(v) and pin_w is not None:
            p_held = self.cluster_pos(self.clusters[held])
            axis = pin_w - p_held
            L = float(np.linalg.norm(axis))
            if L > 1e-6:
                u = axis / L
                t = (v - p_held) @ u
                seg = v[(t > 0.0) & (t < L)]
                # THE cut site: a fixed distance along the appendix from the
                # grasp, toward the body (`resect_offset_m`, 10 mm) -- not the
                # middle of the whole remaining segment, which drifts with
                # wherever the left hand happened to take hold and can sit
                # halfway to the caecum. Snapped to the nearest live surface
                # vertex so it is a point ON the organ, and it tracks the
                # tissue as the left hand lifts it.
                q = p_held + u * self.cfg["resect_offset_m"]
                if len(seg):
                    q = seg[int(np.argmin(np.linalg.norm(seg - q, axis=1)))]
                out["resect_target_rig"] = self._to_rig(q).astype(np.float32)
                out["resect_offset_m"] = np.float32(
                    float(np.dot(q - p_held, u)))
        rig_r = self.rigs.get("right")
        if seg is not None and len(seg) and rig_r is not None:
            out["resect_seg_n"] = np.int32(len(seg))
            out["resect_seg_mid_rig"] = self._to_rig(
                seg.mean(axis=0)).astype(np.float32)
            out["right_tip_to_resect_m"] = np.float32(
                float(np.linalg.norm(seg - rig_r.p_tip_m, axis=1).min()))
        else:
            out["resect_seg_n"] = np.int32(0)
            out["right_tip_to_resect_m"] = np.float32(np.nan)

        # is there appendix actually BETWEEN the blades, and did they close
        # through it (see ArmRig._step_cut -- recorded, never severed)
        if rig_r is not None:
            out["right_tissue_in_jaw"] = bool(rig_r.tissue_near)
            out["right_jaw_tissue_m"] = np.float32(rig_r.probe_thick)
            out["cut_done"] = bool(rig_r.cut_done)
            out["cut_count"] = np.int32(rig_r.n_cuts)
            # WHERE it was cut, on the same 0..1 axis as grasp_frac, so the
            # scorer can require the cut to fall between the grasp and the base
            cfrac = float("nan")
            if rig_r.cut_site is not None and rig_r.cut_done and K:
                d = [float(np.linalg.norm(self.cluster_pos(c) - rig_r.cut_site))
                     for c in self.clusters]
                cfrac = (int(np.argmin(d)) + 0.5) / K
            out["cut_frac"] = np.float32(cfrac)
        out["appendix_severed"] = False   # never: PhysX cannot re-mesh a
        #                                   cooked deformable body

    def _arm_obs(self, rig):
        x, y, z = rig.tool_axes()
        R_tool_w = np.column_stack([x, y, z])
        T_w_tool = _T(R_tool_w, rig.p_tip_m)
        T_rig_tool = self.T_rig_world @ T_w_tool
        r4 = rig.rcm4()
        jaw = (rig.grip_meas if rig.jidx else np.zeros(2))
        opened = (float(np.mean(jaw)) / rig.grip_half_max * 100.0
                  if rig.grip_half_max > 0 else 0.0)
        return dict(
            tip_pose_rig=T_rig_tool.astype(np.float32),
            tip_pose_world=T_w_tool.astype(np.float32),
            tip_pos_rig=T_rig_tool[:3, 3].astype(np.float32),
            tip_quat_rig=_R_to_quat(T_rig_tool[:3, :3]).astype(np.float32),
            # the INSERTION direction (into the patient) = -tip z, so that a
            # policy does not have to remember the sign convention to know
            # which way "further in" is
            shaft_dir_rig=(self.T_rig_world[:3, :3].T
                           @ (rig.a / np.linalg.norm(rig.a))).astype(np.float32),
            insertion_m=np.float32(r4[0]),
            rcm4=r4.astype(np.float32),
            port_id=int(rig.port_no),
            port_pos_rig=(self.T_rig_world[:3, :3].T
                          @ (rig.p_rcm - self.T_world_rig[:3, 3])
                          ).astype(np.float32),
            # what the policy ASKED for, in the same frame -- the gap between
            # this and tip_pos_rig is the arm telling you it could not comply
            tip_cmd_pos_rig=self._to_rig(rig.p_cmd).astype(np.float32),
            cmd_error_m=np.float32(np.linalg.norm(rig.p_cmd - rig.p_tip_m)),
            # shaft roll the action has asked for, what the joint has taken,
            # and what is still owed. A saturated roll is now LATE, not lost --
            # backlog is the visible evidence of that lateness
            roll_cmd_rad=np.float32(rig.roll_cmd),
            roll_meas_rad=np.float32(rig.roll_put),
            roll_backlog_rad=np.float32(rig.roll_cmd - rig.roll_put),
            gripper_cmd=np.float32(rig.grip_cmd_frac * 100.0),
            gripper_meas=np.float32(opened),
            jaw_rad=np.asarray(jaw, np.float32),
            joint_pos=rig.qs.astype(np.float32),
            joint_vel=rig.qd.astype(np.float32),
            grasped=bool(rig.grasped),
            rcm_err_m=np.float32(rig.rcm_err),
            trocar_bite_m=np.float32(rig.bite),
            track_lag_m=np.float32(np.linalg.norm(rig.lag)),
        )

    def _force_obs(self, rig):
        """The instrumented LEFT gripper, in the URDF tool_tip frame.

        bend_x / bend_y  N, lateral force the tissue applies to the instrument
                         along tip +x (the jaw CLOSING direction) and tip +y
                         (the clevis PIN axis). Positive = the tissue pushes
                         the tool along that axis.
        axial            N, along tip -Z, i.e. along the INSERTION direction.
                         Tissue resists being pushed into, so pressing the tip
                         into it makes this NEGATIVE.
        torsion          N.m about tip -Z, the same insertion axis.
        grip             N, jaw closing force at the blade.

        The two shaft channels are referenced to -z, not +z, because -z is the
        direction the instrument points and is the axis an instrument's own
        push/twist is naturally quoted about. `wrench_tool` stays in the plain
        tip frame, so `axial == -wrench_tool[2]` and `torsion == -wrench_tool[5]`
        exactly -- cross-check against it whenever the sign is in doubt.
        """
        w = rig.read_wrench()
        g = rig.read_grip_force()
        m = rig.wrench_model or {}
        n_avg = max(1, self.n_sub)
        return dict(
            bend_x=np.float32(w[0]), bend_y=np.float32(w[1]),
            axial=np.float32(-w[2]),     # along tip -z = the insertion axis
            torsion=np.float32(-w[5]),   # about the same -z
            grip=np.float32(g),
            grip_raw=np.float32(rig.grip_force_raw),
            wrench_tool=w.astype(np.float32),
            contact=bool(rig.tissue_near or rig.grasped),
            latched=bool(rig.grasped),
            valid=bool(rig.row_tool is not None and rig.wrench_model),
            # the identification residual over sqrt(N) of averaging: the
            # smallest change in this channel that means anything at all
            noise_force_n=np.float32(m.get("rms_f", 0.0) / math.sqrt(n_avg)),
            noise_torque_nm=np.float32(m.get("rms_t", 0.0) / math.sqrt(n_avg)),
            noise_grip_n=np.float32(m.get("rms_jaw", 0.0)
                                    / math.sqrt(n_avg)
                                    / max(self.cfg["jaw_length"], 1e-6)),
        )

    # ---- housekeeping ----------------------------------------------------
    def summary(self):
        out = {}
        for nm, rig in self.rigs.items():
            out[nm] = dict(rcm_err_mm=rig.rcm_err * 1000.0,
                           bite_mm=rig.bite * 1000.0,
                           depth_clamped=rig.n_depth_clamped,
                           cone_clamped=rig.n_cone_cmd,
                           grasped=rig.grasped)
        return out

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._timeline.stop()
        except Exception:
            pass
        self.app.close()


# ==========================================================================
# self-test:  python.sh isaac/vla_env.py --steps 40
# ==========================================================================
class DemoPolicy:
    """A hand-written closed-loop baseline -- NOT a learned policy, and it
    cheats: it reads `obs["task"]`, which no real endoscope can see. It exists
    so that the whole pipeline (action -> physics -> every observation channel)
    can be exercised and scored end to end before a real model is plugged in.

    It runs the WHOLE appendectomy, both hands, as an explicit state machine,
    because that is what the five scored stages ask for (see
    vla_rollout.Episode.STAGES) and a baseline that stops after the lift can
    only ever score 2 of them:

        LEFT   standoff -> descend -> close -> lift -> hold ...
        RIGHT  ... wait for the lift, then approach the stretch of appendix
               between the grasp and the body, open the blades around it,
               close through it, withdraw.

    The right hand deliberately waits. Bringing the cutter in before the left
    has the tissue up means cutting against a slack organ lying on its bed, and
    the stage order in the scorer enforces the same sequence, so a baseline
    that raced ahead would fail stages it had physically performed.
    """

    STANDOFF, DESCEND, CLOSE, LIFT, HOLD, RELEASE = range(6)
    (R_WAIT, R_APPROACH, R_ROLL, R_ENGAGE, R_CUT, R_WITHDRAW,
     R_DONE) = range(7)
    # Roll the cutter about ITS OWN SHAFT before closing, so the blades meet
    # the appendix across its long axis instead of along it -- which is how
    # scissors are presented to a tubular structure. Positive is
    # counter-clockwise in the scope image; flip the sign if the scope is ever
    # re-clocked. The rate is deliberately low: the roll is one DOF of a
    # 7-joint null-space solve, and driving it fast makes the wrist chase the
    # target through a large joint excursion while the tip is next to tissue.
    CUT_ROLL_DEG = 90.0
    CUT_ROLL_DPS = 15.0      # deg/s of shaft roll
    SPEED = 0.012            # m/s of commanded tip motion
    LIFT_M = 0.015           # how far to raise the grasped tissue
    STANDOFF_M = 0.010       # hover this far above the handle before descending
    PIN_CLEARANCE_M = 0.025  # keep this far from the appendix's world anchor
    GRASP_CLOSE_S = 2.5      # s for the grasper to close, as a command ramp
    GRASP_SETTLE_S = 1.5     # s held shut before lifting, so the latch can arm
    # The mouth is the 21 mm BEHIND the tip (pin -> tip), so a tip parked on
    # the tissue leaves it in front of the blades, not between them. Drive the
    # tip this far past the target along the shaft and the tissue sits mid-mouth.
    MOUTH_DEPTH_M = 0.011
    CUT_SPEED = 0.008        # m/s: the cutter approaches slower than the grasper
    LIFT_READY_M = 0.008     # m of lift before the right hand is allowed in
    # The scissor action, as a COMMAND ramp. The env slews the jaws at
    # gripper_rate_dps (250 deg/s), so a stepped command opens and shuts them
    # inside a couple of video frames -- these make the gesture legible.
    CUT_CLOSE_S = 2.5        # s to go from fully open to shut
    CUT_HOLD_S = 1.0         # s held shut at the bottom of the stroke
    CUT_OPEN_S = 2.0         # s to reopen, before withdrawing

    def __init__(self, env):
        self.env = env
        self.reset(None)

    def reset(self, obs):
        self.phase = self.STANDOFF
        self.t_phase = 0.0
        self.z0 = 0.0
        self.rphase = self.R_WAIT
        self.t_rphase = 0.0
        self.left_ready = False  # the left hand has lifted and still holds
        self.roll_ref = None     # tool +x captured when the roll began

    def __call__(self, obs):
        env = self.env
        hz = env.cfg["control_hz"]
        chunk = env.cfg["chunk"]
        a = Action.zeros(chunk)
        T = np.asarray(obs["left"]["tip_pose_rig"], np.float64)
        R, p = T[:3, :3], T[:3, 3]
        task = obs["task"]
        handles = task.get("handles_rig")
        free = task.get("handle_free")
        d_pin = task.get("handle_to_pin_m")
        target = None
        if handles is not None and len(handles):
            cand = [(h, (d_pin[i] if d_pin is not None else 1.0))
                    for i, (h, f) in enumerate(zip(handles, free)) if f]
            if not cand:
                cand = [(h, 1.0) for h in handles]
            # Prefer a handle that is NOT sitting on the appendix's world pin.
            # The far end is welded to the world, so lifting the handle next to
            # it pulls against a rigid anchor: measured, that is what threw the
            # tool at load onset, saturated the command leash at 15.7 mm and
            # put the shaft 16 mm off a 3 mm port on 6 of 60 control steps.
            # Among the handles clear of the pin, take the nearest.
            clear = [c for c in cand if c[1] > self.PIN_CLEARANCE_M]
            pool = clear or cand
            target = min(pool, key=lambda c: float(np.linalg.norm(c[0] - p)))[0]
        d = np.zeros(3) if target is None else (target - p)
        grip = 100.0

        # Come down ONTO the handle from a hover, rather than driving straight
        # at it: the handle sits inside the organ, so a direct approach ploughs
        # the tip through the tissue (measured 22 mm of disturbance and a
        # 0.26 mm closest approach, i.e. buried) before the jaws ever close.
        if self.phase == self.STANDOFF:
            hover = (target + np.array([0.0, 0.0, self.STANDOFF_M])
                     if target is not None else p)
            d = hover - p
            if float(np.linalg.norm(d)) < 0.003 or self.t_phase > 12.0:
                self.phase, self.t_phase = self.DESCEND, 0.0
        elif self.phase == self.DESCEND:
            if float(np.linalg.norm(d)) < 0.004 or self.t_phase > 4.0:
                self.phase, self.t_phase = self.CLOSE, 0.0
        elif self.phase == self.CLOSE:
            # Close on a ramp, same reason as the cutter: the env slews the
            # jaws at gripper_rate_dps, so a stepped command shuts them inside
            # a couple of video frames and the grasp is invisible.
            #
            # The DWELL at the bottom is not cosmetic. The latch only arms once
            # the command is actually at zero and the jaws have settled ON the
            # tissue at the force floor (`_step_grasp`), so the ramp has to be
            # followed by a real hold or the grasp would be attempted while the
            # jaws were still travelling.
            grip = float(np.clip(
                100.0 * (1.0 - self.t_phase / self.GRASP_CLOSE_S), 0.0, 100.0))
            d = np.zeros(3)
            if self.t_phase > self.GRASP_CLOSE_S + self.GRASP_SETTLE_S:
                self.phase, self.t_phase = self.LIFT, 0.0
                self.z0 = p[2]
        elif self.phase == self.LIFT:
            grip = 0.0
            d = np.array([0.0, 0.0, self.z0 + self.LIFT_M - p[2]])
            if abs(d[2]) < 0.002 or self.t_phase > 8.0:
                self.phase, self.t_phase = self.HOLD, 0.0
        elif self.phase == self.HOLD:
            # Hold until the CUTTER is finished, not for a fixed couple of
            # seconds. The surgeon's left hand keeps the appendix up for the
            # whole resection, and the scorer reads it the same way -- stages
            # 3-5 all require the lift to still be on, so a left hand that
            # let go on a timer failed jaw_on_tissue and cut having physically
            # performed both (measured: cutter 0.23 mm from the segment, a cut
            # logged at 0.79 along the organ, stage still false).
            #
            # And it never releases. The specimen stays held after the cut
            # (user) -- which is also the real gesture: the resected appendix
            # is lifted out in the grasper, not dropped back into the abdomen.
            # RELEASE is left in the enum for the single-arm diagnostics that
            # still drive it.
            grip, d = 0.0, np.zeros(3)
        else:
            grip, d = 100.0, np.zeros(3)

        # rate-limit, then take the rig-frame step into the TOOL frame, which
        # is the frame the action is defined in
        step = self.SPEED / hz
        n = float(np.linalg.norm(d))
        d = d * (min(step, n) / n) if n > 1e-9 else np.zeros(3)
        d_tool = R.T @ d
        a[Action.LEFT, :, 0:3] = d_tool
        a[Action.LEFT, :, 6] = grip
        self._step_right(obs, task, a, hz)
        self.t_phase += chunk / hz
        self.t_rphase += chunk / hz
        return a

    def _rolled(self, R, axis):
        """Degrees the shaft has rolled since `roll_ref` was captured.

        `axis` must be the axis the COMMAND turns about -- the tool frame's +z,
        not the tip-ward shaft direction `fwd = -z`. Measuring about `fwd`
        reports the opposite sign, so the error `CUT_ROLL_DEG - done` grows as
        the arm rolls instead of shrinking: the stop never fires and the turn
        runs until the phase times out (measured, 184 deg).

        Both the reference and the current jaw-closing axis are projected onto
        the plane perpendicular to the shaft, so pivoting the shaft does not
        read as roll."""
        def perp(v):
            w = v - axis * float(np.dot(v, axis))
            n = float(np.linalg.norm(w))
            return w / n if n > 1e-9 else w
        r0, r1 = perp(self.roll_ref), perp(R[:, 0])
        return math.degrees(math.atan2(float(np.dot(axis, np.cross(r0, r1))),
                                       float(np.dot(r0, r1))))

    def _step_right(self, obs, task, a, hz):
        """The cutter's half of the state machine."""
        T = np.asarray(obs["right"]["tip_pose_rig"], np.float64)
        R, p = T[:3, :3], T[:3, 3]
        # the shaft direction, tip-ward, in the rig frame. The tool frame has
        # +x jaw-closing, +y clevis pin and the shaft along -z (see tool_axes),
        # so advancing the tip DEEPER is a step along -z.
        fwd = -R[:, 2]
        # the point 10 mm along the appendix from the grasp, toward the body
        seg = task.get("resect_target_rig")
        # LATCHED, not instantaneous. Soft tissue relaxes after the lift, so
        # the raw height dips back under LIFT_READY_M and the cutter fell all
        # the way back to `wait` mid-approach, over and over -- measured: it
        # reached the site, bounced, and never rolled or cut. What actually
        # matters is that the left hand HAS lifted and is still holding, so the
        # flag is set once and only cleared by losing the grasp.
        if float(task.get("appendix_lift_m", 0.0)) >= self.LIFT_READY_M:
            self.left_ready = True
        if not bool(obs["left"].get("grasped", False)):
            self.left_ready = False
        lifted = self.left_ready
        in_jaw = bool(task.get("right_tissue_in_jaw", False))
        grip, d, droll = 100.0, np.zeros(3), 0.0

        if self.rphase == self.R_WAIT:
            # jaws open and out of the way until the left hand has the tissue
            grip, d = 30.0, np.zeros(3)
            if lifted and seg is not None and self.t_rphase > 1.0:
                self.rphase, self.t_rphase = self.R_APPROACH, 0.0
        elif self.rphase == self.R_APPROACH:
            # blades WIDE open, aiming past the tissue so it ends up in the
            # mouth rather than in front of the tip
            grip = 100.0
            if seg is not None:
                d = (np.asarray(seg, np.float64)
                     + fwd * self.MOUTH_DEPTH_M) - p
            if seg is None or not lifted:
                self.rphase, self.t_rphase = self.R_WAIT, 0.0
            elif in_jaw or float(np.linalg.norm(d)) < 0.002 \
                    or self.t_rphase > 14.0:
                self.rphase, self.t_rphase = self.R_ROLL, 0.0
        elif self.rphase == self.R_ROLL:
            # square the blades across the appendix, station-keeping while the
            # shaft turns. Blades stay open, so the roll sweeps through tissue
            # rather than winding it up.
            #
            # Stopped on the MEASURED roll, not on what was asked for. Counting
            # requests over-rolled by exactly the chunk factor: the command
            # written here fills all K sub-steps of the chunk, so the arm gets
            # K times the per-call increment while a request counter sees one.
            # Measuring closes that loop and any other -- rate limits, joint
            # saturation, the null-space solve giving some of it back.
            grip = 100.0
            if self.roll_ref is None:
                self.roll_ref = R[:, 0].copy()   # tool +x = jaw closing axis
            done = self._rolled(R, R[:, 2])   # the axis RZ turns about
            want = self.CUT_ROLL_DEG - done
            # `want / chunk`, because every row of the chunk is applied: a
            # command of x deg lands as x * chunk. Without the division the
            # smallest step this loop can take is CUT_ROLL_DPS/hz * chunk =
            # 4 deg, four times the +-1 deg exit band, so it could never land
            # inside it -- it hunted 87.5 -> 91.5 -> 87.5 around the target
            # until the phase timed out. That was the ~2 s of shake after the
            # roll, and the reason a 90 deg turn measured 88-92.
            droll = float(np.clip(want / max(1, a.shape[1]),
                                  -self.CUT_ROLL_DPS / hz,
                                  self.CUT_ROLL_DPS / hz))
            if abs(want) < 1.0 or self.t_rphase > 20.0:
                self.rphase, self.t_rphase = self.R_ENGAGE, 0.0
                droll = 0.0
        elif self.rphase == self.R_ENGAGE:
            # Re-converge on the SAME aim point after the roll, rather than
            # feeding a fixed distance along the shaft. Turning the wrist 90 deg
            # moves the tip a little, so this closes that error out; a blind
            # feed would instead drive the tool straight past the target it had
            # just reached.
            #
            # The aim point is the cut site plus MOUTH_DEPTH_M along the shaft,
            # so the tissue ends up in the middle of the mouth rather than at
            # its opening -- the mouth is the 21 mm BEHIND the tip, so a tip
            # parked exactly on the target leaves the appendix in front of the
            # blades.
            grip = 100.0
            if seg is not None:
                d = (np.asarray(seg, np.float64)
                     + fwd * self.MOUTH_DEPTH_M) - p
            if (in_jaw and float(np.linalg.norm(d)) < 0.002) \
                    or self.t_rphase > 10.0:
                self.rphase, self.t_rphase = self.R_CUT, 0.0
        elif self.rphase == self.R_CUT:
            # Close on a RAMP, hold shut, and stay put while doing it.
            #
            # Stepping the command straight to 0 does close the blades -- the
            # measured opening went 100 -> 0 -> 100 -- but the env slews the
            # jaws at gripper_rate_dps (250 deg/s), so the whole scissor action
            # was over in about six video frames. With the blades phasing
            # through the tissue there is no deformation to signal it either,
            # so the cutter read as "only moving, never cutting" (user).
            # Ramping the COMMAND is what makes it visible: the jaws follow the
            # command, and the command is now the slow part.
            grip = float(np.clip(
                100.0 * (1.0 - self.t_rphase / self.CUT_CLOSE_S), 0.0, 100.0))
            d = np.zeros(3)
            if self.t_rphase > self.CUT_CLOSE_S + self.CUT_HOLD_S:
                self.rphase, self.t_rphase = self.R_WITHDRAW, 0.0
        elif self.rphase == self.R_WITHDRAW:
            # open on the same ramp, THEN back off -- withdrawing while still
            # shut would drag the tissue out with the blades
            grip = float(np.clip(
                100.0 * self.t_rphase / self.CUT_OPEN_S, 0.0, 100.0))
            d = (-fwd * 0.02 if self.t_rphase > self.CUT_OPEN_S
                 else np.zeros(3))
            if self.t_rphase > self.CUT_OPEN_S + 3.0:
                self.rphase, self.t_rphase = self.R_DONE, 0.0
        else:
            grip, d = 100.0, np.zeros(3)

        step = self.CUT_SPEED / hz
        n = float(np.linalg.norm(d))
        d = d * (min(step, n) / n) if n > 1e-9 else np.zeros(3)
        a[Action.RIGHT, :, 0:3] = R.T @ d
        # Only the roll about the shaft survives the RCM, and the shaft is the
        # tool frame's z -- the other two rotation channels are reported back
        # as rejected, so they are left at zero rather than asked for.
        #
        # Per CALL, not divided across the chunk -- measured: the rotation
        # channel is not accumulated over sub-steps the way translation is, so
        # `droll` here lands as `droll` per control call. Dividing it by the
        # chunk (tried) delivered exactly 1/8 of the rate, 1.85 deg/s against
        # the 15 asked, and the roll timed out 37 deg into a 90 deg turn.
        #
        # What keeps this honest is that R_ROLL stops on the MEASURED angle
        # (`_rolled`), so the command only has to be fast enough -- it cannot
        # run past the target the way the old request-counting version did
        # (that one asked for 90 and turned a full 180).
        a[Action.RIGHT, :, Action.RZ] = math.radians(droll)
        a[Action.RIGHT, :, Action.GRIP] = grip

    def label(self):
        return (["standoff", "descend", "close", "lift", "hold",
                 "release"][self.phase] + "/" +
                ["wait", "approach", "roll", "engage", "cut", "withdraw",
                 "done"][self.rphase])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=20,
                    help="step() calls to run in the self-test")
    ap.add_argument("--chunk", type=int, default=DEFAULTS["chunk"])
    ap.add_argument("--control-hz", type=float, default=DEFAULTS["control_hz"])
    ap.add_argument("--gui", action="store_true", help="run windowed")
    ap.add_argument("--stereo", action="store_true")
    ap.add_argument("--endo-vignette", type=float,
                    default=DEFAULTS["endo_vignette"],
                    help="scope circle radius / half frame width (1.0 = touches "
                         "the left and right edges, as on the real tower). "
                         "0 disables it and restores the full rectangle")
    ap.add_argument("--dump", default="",
                    help="directory for one PNG per camera per step")
    args = ap.parse_args(argv)

    env = LaparoVLAEnv(headless=not args.gui, chunk=args.chunk,
                       control_hz=args.control_hz, stereo=args.stereo,
                       endo_vignette=args.endo_vignette)
    obs = env.reset()
    _log("--- observation keys -------------------------------------------")
    for k, v in obs.items():
        if isinstance(v, dict):
            _log(f"  {k}:")
            for kk, vv in v.items():
                shape = (getattr(vv, "shape", None) if hasattr(vv, "shape")
                         else None)
                _log(f"      {kk:16s} {shape if shape is not None else vv}")
        else:
            _log(f"  {k}: {v}")
    dump = Path(args.dump) if args.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio
    t0 = time.time()
    f0 = obs["left_force"]
    _log(f"force channel noise floor: {float(f0['noise_force_n'])*1000:.1f} mN "
         f"| {float(f0['noise_torque_nm'])*1000:.2f} mN.m "
         f"| grip {float(f0['noise_grip_n'])*1000:.2f} mN")
    policy = DemoPolicy(env)
    for k in range(args.steps):
        obs, info = env.step(policy(obs))
        f, l, tk = obs["left_force"], obs["left"], obs["task"]
        _log(f"step {k:3d} t={obs['t']:5.2f}s {policy.label():>8s} "
             f"d={float(tk.get('left_tip_to_appendix_m', np.nan))*1000:5.1f}mm "
             f"ins={float(l['insertion_m'])*1000:5.1f}mm "
             f"grip {float(l['gripper_cmd']):5.1f}->{float(l['gripper_meas']):5.1f} "
             f"| F=({float(f['bend_x']):+6.2f},{float(f['bend_y']):+6.2f},"
             f"{float(f['axial']):+6.2f})N tau={float(f['torsion'])*1000:+6.1f}mNm "
             f"grip={float(f['grip'])*1000:6.1f}mN "
             f"| rcm={info['left']['rcm_err_mm']:.2f}mm "
             f"lag={info['left']['track_lag_mm']:.1f}mm "
             f"grasp={int(l['grasped'])} "
             f"lift={float(tk.get('appendix_lift_m', 0.0))*1000:+5.1f}mm")
        if dump:
            for nm in ("endoscope", "d405"):
                if "rgb" in obs[nm]:
                    iio.imwrite(dump / f"{nm}_{k:04d}.png", obs[nm]["rgb"])
            if "depth" in obs["d405"]:
                d = obs["d405"]["depth"]
                m = (d > 0) & np.isfinite(d)
                lo, hi = (float(d[m].min()), float(d[m].max())) if m.any() else (0, 1)
                iio.imwrite(dump / f"d405_depth_{k:04d}.png",
                            (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
                             * 255).astype(np.uint8))
    dt = time.time() - t0
    _log(f"{args.steps} steps in {dt:.1f} s "
         f"({args.steps/max(dt,1e-6):.2f} step/s, "
         f"{args.steps*args.chunk/max(dt,1e-6):.1f} action sub-steps/s)")
    _log(json.dumps(env.summary(), indent=2, default=float))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
