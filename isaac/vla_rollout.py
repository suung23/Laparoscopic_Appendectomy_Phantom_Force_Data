#!/usr/bin/env python3
"""Run a policy in the surgical cell, record the episode, score it, make videos.

    # scripted baseline, one episode, video + npz
    ~/isaacsim-5.1/python.sh isaac/vla_rollout.py --episodes 1 --video

    # your own policy
    ~/isaacsim-5.1/python.sh isaac/vla_rollout.py --policy mypolicy.py:Policy \
        --episodes 20 --seconds 25 --out out/eval_run1 --video

A POLICY is any Python object with

    __call__(obs) -> action        # (2, K, 7) float32, K <= env chunk

optionally plus `reset(obs)`, called once at the start of each episode.
`--policy path/to/file.py:ClassName` imports the file and constructs
`ClassName(env)` if the constructor takes an argument, else `ClassName()`.

Everything the run produced lands under `--out`:

    episode_000.npz         every observation channel, per control step
    episode_000_endoscope.mp4
    episode_000_d405.mp4
    episode_000_d405_depth.mp4
    summary.json            per-episode metrics + the aggregate
    frames/                 the PNGs the videos were built from (--keep-frames)

The .npz stores images as uint8 arrays; a 25 s episode at 30 Hz is 750 frames,
which is ~2 GB of endoscope RGB. `--image-every N` keeps every Nth frame's
images (metrics and proprioception are always kept at full rate) and
`--no-images` drops them entirely -- the videos are still written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def physics_health_gate(state_finite, volume_ratio, inverted_tets,
                        nonfinite_tets):
    """Return True/False when backend health telemetry exists, else None."""
    arrays = [np.asarray(x, dtype=np.float64).reshape(-1) for x in
              (state_finite, volume_ratio, inverted_tets, nonfinite_tets)]
    if not any(np.isfinite(x).any() for x in arrays):
        return None
    finite_state, volumes, inverted, nonfinite = arrays
    return bool(
        finite_state.size and np.isfinite(finite_state).all()
        and np.all(finite_state > 0.5)
        and volumes.size and np.isfinite(volumes).all()
        and np.all(volumes > 0.0)
        and inverted.size and np.isfinite(inverted).all()
        and np.all(inverted == 0.0)
        and nonfinite.size and np.isfinite(nonfinite).all()
        and np.all(nonfinite == 0.0))


# --------------------------------------------------------------------------
# metrics -- what a rollout is scored on
# --------------------------------------------------------------------------
class Episode:
    """Accumulates one episode and turns it into arrays + a scorecard."""

    # scalar channels, flattened out of the nested observation
    SCALARS = {
        "t": ("t",),
        "left_ins": ("left", "insertion_m"),
        "right_ins": ("right", "insertion_m"),
        "left_grip_cmd": ("left", "gripper_cmd"),
        "left_grip_meas": ("left", "gripper_meas"),
        "right_grip_cmd": ("right", "gripper_cmd"),
        "right_grip_meas": ("right", "gripper_meas"),
        "left_rcm_err": ("left", "rcm_err_m"),
        "right_rcm_err": ("right", "rcm_err_m"),
        "left_bite": ("left", "trocar_bite_m"),
        "right_bite": ("right", "trocar_bite_m"),
        "left_lag": ("left", "track_lag_m"),
        "right_lag": ("right", "track_lag_m"),
        "f_bend_x": ("left_force", "bend_x"),
        "f_bend_y": ("left_force", "bend_y"),
        "f_axial": ("left_force", "axial"),
        "f_torsion": ("left_force", "torsion"),
        "f_grip": ("left_force", "grip"),
        "f_grip_raw": ("left_force", "grip_raw"),
        "appendix_lift": ("task", "appendix_lift_m"),
        "appendix_moved": ("task", "appendix_moved_m"),
        "left_tip_to_appendix": ("task", "left_tip_to_appendix_m"),
        "right_tip_to_pin": ("task", "right_tip_to_pin_m"),
        # the staged resection (see LaparoVLAEnv._resection_obs)
        "grasp_frac": ("task", "grasp_frac"),
        "right_tip_to_resect": ("task", "right_tip_to_resect_m"),
        "right_jaw_tissue": ("task", "right_jaw_tissue_m"),
        "cut_frac": ("task", "cut_frac"),
        # Newton-only numerical health channels; absent on the Isaac backend
        # until equivalent FEM telemetry is plumbed through there.
        "physics_state_finite": ("task", "physics_state_finite"),
        "tissue_min_signed_volume_ratio":
            ("task", "tissue_min_signed_volume_ratio"),
        "tissue_inverted_tets": ("task", "tissue_inverted_tets"),
        "tissue_nonfinite_tets": ("task", "tissue_nonfinite_tets"),
        "tissue_max_speed_m_s": ("task", "tissue_max_speed_m_s"),
        # Per-organ FEM telemetry identifies which captured mesh first fails
        # the global signed-volume gate. Backends without these observations
        # leave the channels unavailable (NaN), never implicitly healthy.
        **{f"{organ}_inverted_tets": (
            "task", "tissue_health_by_organ", organ, "tissue_inverted_tets")
           for organ in ("cecum", "bowel1", "bowel2", "bowel3",
                         "stomach", "appendix")},
        **{f"{organ}_min_volume_ratio": (
            "task", "tissue_health_by_organ", organ,
            "tissue_min_signed_volume_ratio")
           for organ in ("cecum", "bowel1", "bowel2", "bowel3",
                         "stomach", "appendix")},
        **{f"{organ}_worst_tet_index": (
            "task", "tissue_health_by_organ", organ,
            "tissue_worst_tet_index")
           for organ in ("cecum", "bowel1", "bowel2", "bowel3",
                         "stomach", "appendix")},
    }
    VECTORS = {
        "left_tip_pose_rig": ("left", "tip_pose_rig"),
        "right_tip_pose_rig": ("right", "tip_pose_rig"),
        "left_rcm4": ("left", "rcm4"),
        "right_rcm4": ("right", "rcm4"),
        "left_joint_pos": ("left", "joint_pos"),
        "right_joint_pos": ("right", "joint_pos"),
        "left_wrench_tool": ("left_force", "wrench_tool"),
    }
    FLAGS = {
        "left_grasped": ("left", "grasped"),
        "right_grasped": ("right", "grasped"),
        "contact": ("left_force", "contact"),
        "grasp_in_band": ("task", "grasp_in_band"),
        "right_tissue_in_jaw": ("task", "right_tissue_in_jaw"),
        "cut_done": ("task", "cut_done"),
    }

    LIFT_SUCCESS_M = 0.010          # tissue raised this far while held = win
    # ...and how high it has to STILL be for the later stages to count. Soft
    # tissue relaxes: the demo run peaked at 11.7 mm and settled to 8.7 mm
    # while the grasp held perfectly, so requiring the peak throughout failed
    # jaw_on_tissue and cut on an episode that performed both (the cutter was
    # 0.24 mm from the resection segment with appendix between its blades).
    # The stage-2 achievement is still the full LIFT_SUCCESS_M; this only says
    # the tissue must remain LIFTED, not that it must not creep.
    LIFT_HOLD_M = 0.006
    RESECTION_APPROACH_M = 0.005    # right tip must get this close to the
                                     # resection segment to be "at the site"
    RESECTION_JAW_CLOSED_PCT = 15.0 # right jaw opening (%) below which the
                                     # tool reads as closing on tissue, not
                                     # just hovering near it
    # An appendectomy is a SEQUENCE, and doing the steps in the wrong order is
    # not doing the task: cutting before the tissue is lifted, or lifting after
    # the cutter has already closed, both score every stage individually while
    # being nothing like the procedure. So each stage is only looked for at or
    # after the step the one before it was first satisfied (`_first_after`),
    # and `success` is all five reached in order on a rig that stayed in its
    # trocars.
    STAGES = ("grasp", "lift", "at_site", "jaw_on_tissue", "cut")

    def __init__(self, idx, keep_images=True, image_every=1):
        self.idx = idx
        self.keep_images = keep_images
        self.image_every = max(1, int(image_every))
        self.rows = {k: [] for k in
                     list(self.SCALARS) + list(self.VECTORS) + list(self.FLAGS)}
        self.actions = []
        self.images = {}
        self.meta = {}
        self.n = 0
        self.violated = ""
        self.physics_failure = ""
        self.physics_failure_details = None
        self.rcm_excursions = 0

    def add(self, obs, action=None):
        for k, path in self.SCALARS.items():
            v = obs
            for p in path:
                v = v.get(p, np.nan) if isinstance(v, dict) else np.nan
            self.rows[k].append(float(v) if np.isscalar(v) or
                                isinstance(v, np.generic) else np.nan)
        for k, path in self.VECTORS.items():
            v = obs
            for p in path:
                v = v.get(p) if isinstance(v, dict) else None
            self.rows[k].append(np.asarray(v, np.float32) if v is not None
                                else None)
        for k, path in self.FLAGS.items():
            v = obs
            for p in path:
                v = v.get(p, False) if isinstance(v, dict) else False
            self.rows[k].append(bool(v))
        if action is not None:
            self.actions.append(np.asarray(action, np.float32))
        if self.keep_images and self.n % self.image_every == 0:
            for cam in ("endoscope", "d405", "d405_right"):
                if cam in obs and "rgb" in obs[cam]:
                    self.images.setdefault(f"{cam}_rgb", []).append(
                        obs[cam]["rgb"])
                if cam in obs and "depth" in obs[cam]:
                    self.images.setdefault(f"{cam}_depth", []).append(
                        obs[cam]["depth"].astype(np.float16))
        if not self.meta:
            self.meta = {
                f"{cam}_K": np.asarray(obs[cam]["K"], np.float32)
                for cam in ("endoscope", "d405") if cam in obs}
            self.meta["T_world_rig"] = np.asarray(obs["T_world_rig"],
                                                  np.float32)
            for cam in ("endoscope", "d405"):
                if cam in obs:
                    self.meta[f"{cam}_T_rig_cam"] = np.asarray(
                        obs[cam]["T_rig_cam"], np.float32)
        self.n += 1

    # -- scoring ----------------------------------------------------------
    @staticmethod
    def _first_after(mask, after):
        """First index >= `after` where `mask` holds, or -1. `after` < 0 (the
        previous stage never happened) short-circuits to -1, which is what
        makes the chain ordered rather than five independent tests."""
        if after < 0:
            return -1
        idx = np.flatnonzero(np.asarray(mask, bool)[after:])
        return int(idx[0] + after) if len(idx) else -1

    def _stages(self, a):
        """The five stages of the task, in order, as first-satisfied indices.

        1 grasp         the LEFT jaws hold the appendix, in the mid-body band
                        -- not the free tip (no longer in the scope view) and
                        not the base (leaves nothing to cut between the grasp
                        and the body)
        2 lift          ...and the tissue is up by LIFT_SUCCESS_M, still held
        3 at_site       the RIGHT tip reaches the appendix BETWEEN that grasp
                        and the base, while the lift holds
        4 jaw_on_tissue appendix is actually between the blades -- the measured
                        mouth geometry, not tip proximity
        5 cut           the blades closed through it, on the base side of the
                        grasp (the appendix is not severed; see _step_cut)
        """
        n = self.n
        z = np.zeros(n, bool)

        def col(k, default=np.nan):
            v = a.get(k)
            return v if v is not None and len(v) == n else np.full(n, default)

        held = np.asarray(self.rows["left_grasped"], bool)
        in_band = np.asarray(self.rows["grasp_in_band"], bool)
        in_jaw = np.asarray(self.rows["right_tissue_in_jaw"], bool)
        cut = np.asarray(self.rows["cut_done"], bool)
        lift = col("appendix_lift", 0.0)
        to_resect = col("right_tip_to_resect")
        gfrac, cfrac = col("grasp_frac"), col("cut_frac")

        good_grasp = held & in_band
        lifted = good_grasp & (lift >= self.LIFT_SUCCESS_M)
        # the cutter stages need the tissue STILL held and STILL up, not still
        # at the peak of the lift -- see LIFT_HOLD_M
        holding = good_grasp & (lift >= self.LIFT_HOLD_M)
        at_site = holding & (to_resect <= self.RESECTION_APPROACH_M)
        # the cut has to land on the BASE side of the grasp: cut_frac and
        # grasp_frac share the 0 (free tip) .. 1 (base) axis, so "between the
        # grasp and the body" is simply a larger fraction.
        cut_ok = cut & np.greater(cfrac, gfrac,
                                  out=z.copy(), where=np.isfinite(cfrac)
                                  & np.isfinite(gfrac))

        i = {}
        i["grasp"] = self._first_after(good_grasp, 0)
        i["lift"] = self._first_after(lifted, i["grasp"])
        i["at_site"] = self._first_after(at_site, i["lift"])
        i["jaw_on_tissue"] = self._first_after(holding & in_jaw, i["at_site"])
        i["cut"] = self._first_after(cut_ok, i["jaw_on_tissue"])
        return i, dict(held=held, good_grasp=good_grasp, lifted=lifted,
                       at_site=at_site, in_jaw=in_jaw, to_resect=to_resect,
                       gfrac=gfrac, cfrac=cfrac)

    def score(self):
        a = {k: np.asarray(v, np.float32) for k, v in self.rows.items()
             if v and not isinstance(v[0], (np.ndarray, type(None)))}
        held = np.asarray(self.rows["left_grasped"], bool)
        lift = a.get("appendix_lift", np.zeros(1))
        lift_held = lift[held] if held.any() else np.zeros(1)
        # resection gesture: RIGHT tool at the base, jaws closing, while the
        # LEFT hand is holding the tissue up -- the two-handed sequence a
        # real appendectomy follows. This does not cut anything (no tissue
        # severing is simulated); it is a proxy for "brought the cutter to
        # the right place at the right time"
        n = len(held)
        lift_ok = held & (lift >= self.LIFT_SUCCESS_M)
        right_to_pin = a.get("right_tip_to_pin", np.full(n, np.nan))
        right_jaw = a.get("right_grip_meas", np.full(n, np.nan))
        at_base = lift_ok & (right_to_pin <= self.RESECTION_APPROACH_M)
        resecting = at_base & (right_jaw <= self.RESECTION_JAW_CLOSED_PCT)
        stage_i, sd = self._stages(a)
        stage_done = {s: stage_i[s] >= 0 for s in self.STAGES}
        health_finite = a.get("physics_state_finite", np.full(n, np.nan))
        health_available = bool(np.isfinite(health_finite).any())
        physics_state_finite = (
            bool(np.all(health_finite[np.isfinite(health_finite)] > 0.5))
            if health_available else None)
        volume_ratio = a.get("tissue_min_signed_volume_ratio", np.full(n, np.nan))
        inverted_tets = a.get("tissue_inverted_tets", np.full(n, np.nan))
        nonfinite_tets = a.get("tissue_nonfinite_tets", np.full(n, np.nan))
        tissue_speed = a.get("tissue_max_speed_m_s", np.full(n, np.nan))
        physics_health_pass = physics_health_gate(
            health_finite, volume_ratio, inverted_tets, nonfinite_tets)
        physics_health_failed = physics_health_pass is False
        rcm = np.concatenate([a["left_rcm_err"], a["right_rcm_err"]])
        bite = np.concatenate([a["left_bite"], a["right_bite"]])
        f = np.stack([a["f_bend_x"], a["f_bend_y"], a["f_axial"]], 1)
        return dict(
            steps=int(self.n),
            duration_s=float(a["t"][-1]) if len(a.get("t", [])) else 0.0,
            grasped_frac=float(held.mean()) if len(held) else 0.0,
            grasp_events=int(np.sum(np.diff(held.astype(int)) > 0))
            if len(held) > 1 else 0,
            max_lift_mm=float(lift_held.max() * 1000.0),
            # a SUSTAINED trocar violation voids the run; transient
            # excursions are reported instead of hidden, because an episode
            # that only succeeded by briefly leaving the port is not a result
            trocar_violation=self.violated,
            rcm_excursion_steps=int(self.rcm_excursions),
            rcm_excursion_frac=float(self.rcm_excursions / max(self.n - 1, 1)),
            # SUCCESS IS THE WHOLE PROCEDURE. It used to be grasp+lift, with
            # the resection tracked alongside as a separate, additive flag.
            # The task is the appendectomy, and an episode that grabs and
            # lifts but never brings the cutter to the right stretch of
            # appendix has not done it (user), so all five stages have to land,
            # in order, on a rig that stayed in its trocars.
            success=bool(stage_done["cut"] and not self.violated
                         and not physics_health_failed),
            physics_health_available=health_available,
            physics_state_finite=physics_state_finite,
            physics_health_pass=physics_health_pass,
            physics_failure=self.physics_failure,
            physics_failure_details=self.physics_failure_details,
            minimum_signed_volume_ratio=(
                float(np.nanmin(volume_ratio)) if np.isfinite(volume_ratio).any()
                else float("nan")),
            maximum_inverted_tetrahedra=(
                int(np.nanmax(inverted_tets)) if np.isfinite(inverted_tets).any()
                else None),
            maximum_nonfinite_tetrahedra=(
                int(np.nanmax(nonfinite_tets)) if np.isfinite(nonfinite_tets).any()
                else None),
            peak_tissue_speed_m_s=(
                float(np.nanmax(tissue_speed)) if np.isfinite(tissue_speed).any()
                else float("nan")),
            # the old definition, kept because it is what earlier runs were
            # scored on and the two are worth comparing
            grasp_lift_success=bool(held.any()
                                    and lift_held.max() >= self.LIFT_SUCCESS_M
                                    and not self.violated
                                    and not physics_health_failed),
            stages_completed=int(sum(stage_done.values())),
            stages_total=len(self.STAGES),
            task_progress=float(sum(stage_done.values()) / len(self.STAGES)),
            # which stage it stopped at -- the single most useful field when a
            # run fails, and the one a bare success rate hides
            failed_stage=next((s for s in self.STAGES if not stage_done[s]),
                              ""),
            **{f"stage_{s}": bool(stage_done[s]) for s in self.STAGES},
            **{f"stage_{s}_step": int(stage_i[s]) for s in self.STAGES},
            **{f"stage_{s}_t_s": (float(a["t"][stage_i[s]])
                                  if stage_i[s] >= 0 and len(a.get("t", []))
                                  else float("nan"))
               for s in self.STAGES},
            # where on the appendix each hand worked, 0 = free tip, 1 = base
            grasp_frac=float(np.nanmedian(sd["gfrac"][sd["held"]]))
            if sd["held"].any() and np.isfinite(sd["gfrac"][sd["held"]]).any()
            else float("nan"),
            grasp_in_band_frac=float(sd["good_grasp"].mean()) if n else 0.0,
            cut_frac=float(np.nanmax(sd["cfrac"]))
            if np.isfinite(sd["cfrac"]).any() else float("nan"),
            cut_count=int(np.asarray(self.rows["cut_done"], bool).any()),
            min_right_tip_to_resect_mm=float(
                np.nanmin(sd["to_resect"]) * 1000.0)
            if np.isfinite(sd["to_resect"]).any() else float("nan"),
            jaw_on_tissue_frac=float(sd["in_jaw"].mean()) if n else 0.0,
            max_jaw_tissue_mm=float(np.nanmax(a["right_jaw_tissue"]) * 1000.0)
            if "right_jaw_tissue" in a
            and np.isfinite(a["right_jaw_tissue"]).any() else float("nan"),
            # the appendix is NEVER severed -- PhysX cannot re-mesh a cooked
            # deformable, so stage 5 records the closing gesture through
            # tissue, it does not divide it. Stated here so a reader of
            # summary.json cannot mistake the flag for a cut organ.
            appendix_severed=False,
            # the older base-approach proxy, unchanged
            resection_gesture=bool(resecting.any()),
            resection_events=int(np.sum(np.diff(resecting.astype(int)) > 0))
            if len(resecting) > 1 else 0,
            resection_approach_frac=float(at_base.mean()) if n else 0.0,
            min_right_tip_to_pin_mm=float(np.nanmin(right_to_pin) * 1000.0)
            if n and np.isfinite(right_to_pin).any() else float("nan"),
            resection_success=bool(stage_done["cut"] and not self.violated),
            min_tip_to_appendix_mm=float(
                np.nanmin(a["left_tip_to_appendix"]) * 1000.0)
            if "left_tip_to_appendix" in a else float("nan"),
            tissue_disturbance_mm=float(np.nanmax(a["appendix_moved"]) * 1000.0)
            if "appendix_moved" in a else float("nan"),
            # rig health -- an episode that scores well while sawing the
            # trocar open has not actually succeeded
            rcm_err_p95_mm=float(np.percentile(rcm, 95) * 1000.0),
            rcm_err_max_mm=float(rcm.max() * 1000.0),
            trocar_bite_p95_mm=float(np.percentile(bite, 95) * 1000.0),
            trocar_on_rubber_frac=float((bite > 0).mean()),
            track_lag_p95_mm=float(np.percentile(
                np.concatenate([a["left_lag"], a["right_lag"]]), 95) * 1000.0),
            peak_tip_force_n=float(np.linalg.norm(f, axis=1).max()),
            peak_grip_force_mn=float(a["f_grip"].max() * 1000.0),
            peak_torsion_mnm=float(np.abs(a["f_torsion"]).max() * 1000.0),
        )

    @staticmethod
    def _stack(name, v):
        """Stack a channel, dropping frames that do not conform.

        A channel can go ragged for real reasons -- an annotator that has not
        published yet returns no image at all, an observation key that only
        appears once the tissue is cooked -- and losing the whole episode's
        .npz at save time to one odd frame is the wrong trade. What is dropped
        is named."""
        v = [x for x in v if x is not None]
        if not v:
            return None
        shapes = {np.shape(x) for x in v}
        if len(shapes) > 1:
            keep = max(shapes, key=lambda s: sum(np.shape(x) == s for x in v))
            n0 = len(v)
            v = [x for x in v if np.shape(x) == keep]
            print(f"  note: '{name}' had {len(shapes)} different shapes "
                  f"{sorted(shapes)}; kept the {len(v)}/{n0} frames of "
                  f"shape {keep}")
        return np.stack(v)

    def save(self, path: Path):
        out = {}
        for k, v in self.rows.items():
            if not v:
                continue
            if isinstance(v[0], np.ndarray):
                s = self._stack(k, v)
                if s is not None:
                    out[k] = s.astype(np.float32)
            else:
                out[k] = np.asarray(v)
        if self.actions:
            s = self._stack("action", self.actions)
            if s is not None:
                out["action"] = s
        for k, v in self.images.items():
            s = self._stack(k, v)
            if s is not None:
                out[k] = s
        for k, v in self.meta.items():
            out[f"meta_{k}"] = v
        out["image_every"] = np.int32(self.image_every)
        np.savez_compressed(path, **out)
        return path


# --------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------
def depth_to_rgb(d, lo=None, hi=None):
    """Turquoise->yellow ramp, near = bright. Invalid (0) stays black."""
    d = np.asarray(d, np.float32)
    m = np.isfinite(d) & (d > 1e-4)
    if not m.any():
        return np.zeros(d.shape + (3,), np.uint8)
    lo = float(np.percentile(d[m], 2)) if lo is None else lo
    hi = float(np.percentile(d[m], 98)) if hi is None else hi
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t = 1.0 - t                                     # near = high
    rgb = np.stack([np.clip(1.6 * t - 0.35, 0, 1),
                    np.clip(1.35 * t, 0, 1),
                    np.clip(1.15 - 1.5 * t, 0, 1)], -1)
    rgb[~m] = 0.0
    return (rgb * 255).astype(np.uint8)


def write_video(frames_dir: Path, pattern: str, out_mp4: Path, fps: float):
    """PNG sequence -> H.264. Uses the system ffmpeg (imageio-ffmpeg is not
    installed in the Isaac python)."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        print(f"  ffmpeg not found -- leaving the PNGs in {frames_dir}")
        return None
    cmd = [ff, "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
           "-i", str(frames_dir / pattern), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "20",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed: {r.stderr.strip()[:400]}")
        return None
    return out_mp4


# --------------------------------------------------------------------------
# policy loading
# --------------------------------------------------------------------------
def load_policy(spec, env):
    """`path/to/file.py:ClassName` -> an instance. `builtin:demo` -> the
    scripted baseline that ships with the env."""
    if spec in ("", "builtin:demo", "demo"):
        from vla_env import DemoPolicy
        return DemoPolicy(env)
    if ":" not in spec:
        raise SystemExit("--policy must be 'file.py:ClassName' or 'demo'")
    path, cls = spec.rsplit(":", 1)
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"policy file not found: {path}")
    sys.path.insert(0, str(path.parent))
    mod_spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    if not hasattr(mod, cls):
        raise SystemExit(f"{path} has no attribute {cls!r}")
    obj = getattr(mod, cls)
    try:
        return obj(env)
    except TypeError:
        return obj()


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", default="demo",
                    help="'demo' or path/to/file.py:ClassName")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="simulated seconds per episode")
    ap.add_argument("--out", default="out/vla_rollout")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--control-hz", type=float, default=30.0)
    ap.add_argument("--physics-hz", type=float, default=480.0)
    ap.add_argument("--video", action="store_true", help="write mp4s")
    ap.add_argument("--video-fps", type=float, default=0.0,
                    help="0 = real time (control_hz / image_every)")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--no-images", action="store_true",
                    help="do not store images in the npz (videos still work)")
    ap.add_argument("--continue-invalid-physics", action="store_true",
                    help="diagnostic only: continue after invalid tissue physics; "
                         "episode remains failed")
    ap.add_argument("--image-every", type=int, default=1)
    ap.add_argument("--stereo", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # None = keep the env's DEFAULTS, so the shipped level lives in exactly one
    # place. Pass --rcm-abort-mm 3 to get the pre-2026-08-13 behaviour back.
    ap.add_argument("--rcm-abort-mm", type=float, default=None,
                    help="RCM error that ends the episode after "
                         "--rcm-abort-steps consecutive control steps "
                         "(default: the env's 10 mm)")
    ap.add_argument("--rcm-excursion-mm", type=float, default=None,
                    help="RCM error counted as leaving the port in "
                         "rcm_excursion_frac/_steps. Independent of the abort; "
                         "default 3 mm, the physical 6 mm/5 mm budget")
    ap.add_argument("--rcm-abort-steps", type=int, default=None,
                    help="consecutive steps above --rcm-abort-mm before the "
                         "episode is aborted (default 5)")
    args = ap.parse_args(argv)

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_root = out / "frames"

    from vla_env import LaparoVLAEnv
    over = {k: v for k, v in (("rcm_abort_mm", args.rcm_abort_mm),
                              ("rcm_excursion_mm", args.rcm_excursion_mm),
                              ("rcm_abort_steps", args.rcm_abort_steps))
            if v is not None}
    env = LaparoVLAEnv(headless=not args.gui, chunk=args.chunk,
                       control_hz=args.control_hz, physics_hz=args.physics_hz,
                       stereo=args.stereo, **over)
    steps = max(1, int(round(args.seconds * args.control_hz / args.chunk)))
    print(f"{args.episodes} episode(s) x {steps} step(s) of {args.chunk} "
          f"sub-steps = {args.seconds:.1f} s of simulated time each")

    import imageio.v3 as iio
    # constructed ONCE: a policy that carries learned state across episodes
    # (or just a loaded checkpoint) must not be rebuilt every reset
    policy = load_policy(args.policy, env)
    results = []
    t_run = time.time()
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        if hasattr(policy, "reset"):
            policy.reset(obs)
        rec = Episode(ep, keep_images=not args.no_images,
                      image_every=args.image_every)
        fdir = frames_root / f"ep{ep:03d}"
        if args.video:
            if fdir.exists():
                shutil.rmtree(fdir)
            fdir.mkdir(parents=True, exist_ok=True)
        rec.add(obs)
        nvid = 0
        t0 = time.time()
        for k in range(steps):
            act = policy(obs)
            obs, info = env.step(act)
            rec.add(obs, act)
            if info.get("physics_invalid"):
                rec.physics_failure = str(info["physics_invalid"])
                rec.physics_failure_details = info.get(
                    "physics_contact_diagnostics")
                print(f"    episode {ep} STOPPED at step {k}: invalid physics: "
                      f"{rec.physics_failure}", flush=True)
                if not args.continue_invalid_physics:
                    break
            if args.video and k % args.image_every == 0:
                if "rgb" in obs["endoscope"]:
                    iio.imwrite(fdir / f"endoscope_{nvid:05d}.png",
                                obs["endoscope"]["rgb"])
                if "rgb" in obs["d405"]:
                    iio.imwrite(fdir / f"d405_{nvid:05d}.png",
                                obs["d405"]["rgb"])
                if "depth" in obs["d405"]:
                    iio.imwrite(fdir / f"d405depth_{nvid:05d}.png",
                                depth_to_rgb(obs["d405"]["depth"]))
                if "rgb_right" in obs["d405"] and obs["d405"]["rgb_right"] is not None:
                    iio.imwrite(fdir / f"d405right_{nvid:05d}.png",
                                obs["d405"]["rgb_right"])
                nvid += 1
            rec.rcm_excursions = int(info.get("rcm_violation_total", 0))
            if info.get("terminated"):
                # SUSTAINED: the shaft has been outside its 6 mm port for
                # rcm_abort_steps in a row and is not coming back, so nothing
                # measured after this corresponds to anything the real rig
                # could do. A one- or two-step excursion at load onset does
                # NOT land here -- it is counted in rcm_excursions instead.
                rec.violated = ",".join(info["rcm_violation"]) or "left/right"
                # the Isaac env carries cfg as a dict, the Newton one as a
                # namespace, and this runner serves both
                lvl = (env.cfg.get("rcm_abort_mm") if isinstance(env.cfg, dict)
                       else getattr(env.cfg, "rcm_abort_mm", None))
                print(f"    episode {ep} ABORTED at step {k}: sustained "
                      f"trocar violation on {rec.violated} "
                      f"({info['rcm_violation_steps']} consecutive steps"
                      + (f" over {lvl:g} mm)" if lvl is not None else ")"))
                break
            if info.get("truncated"):
                break
        wall = time.time() - t0
        sc = rec.score()
        sc["episode"] = ep
        sc["wall_s"] = round(wall, 1)
        sc["realtime_factor"] = round(sc["duration_s"] / max(wall, 1e-6), 3)
        results.append(sc)
        npz = rec.save(out / f"episode_{ep:03d}.npz")
        # the stage ladder, so a failure says WHERE it stopped rather than
        # only that it did
        ladder = " ".join(("[+]" if sc[f"stage_{s}"] else "[ ]") + s
                          for s in Episode.STAGES)
        print(f"episode {ep}: {'SUCCESS' if sc['success'] else 'fail    '} "
              f"{sc['stages_completed']}/{sc['stages_total']}  {ladder}"
              + (f"  (stopped at {sc['failed_stage']})"
                 if sc["failed_stage"] else ""))
        print(f"           lift {sc['max_lift_mm']:+5.1f} mm, grasp "
              f"{sc['grasp_events']}x ({sc['grasped_frac']*100:.0f}% of steps) "
              f"at {sc['grasp_frac']:.2f} along the appendix, "
              f"cutter reached {sc['min_right_tip_to_resect_mm']:.1f} mm of "
              f"the resection segment, tissue in the blades "
              f"{sc['jaw_on_tissue_frac']*100:.0f}% of steps"
              + (f", CUT at {sc['cut_frac']:.2f}" if sc["stage_cut"] else "")
              + ", ")
        print(f"           peak tip force {sc['peak_tip_force_n']:.2f} N, "
              f"grip {sc['peak_grip_force_mn']:.0f} mN, "
              f"RCM p95 {sc['rcm_err_p95_mm']:.2f} mm -> {npz.name} "
              f"[{wall:.0f} s wall, {sc['realtime_factor']:.2f}x real time]")
        if args.video:
            fps = args.video_fps or (args.control_hz / args.image_every)
            for tag, pat in (("endoscope", "endoscope_%05d.png"),
                             ("d405", "d405_%05d.png"),
                             ("d405_depth", "d405depth_%05d.png"),
                             ("d405_right", "d405right_%05d.png")):
                if not list(fdir.glob(pat.split("_%")[0] + "_*.png")):
                    continue
                mp4 = write_video(fdir, pat,
                                  out / f"episode_{ep:03d}_{tag}.mp4", fps)
                if mp4:
                    print(f"    {mp4.relative_to(out.parent)}")
            if not args.keep_frames:
                shutil.rmtree(fdir, ignore_errors=True)

    agg = {}
    if results:
        keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]
        agg = {k: float(np.mean([r[k] for r in results])) for k in keys}
        agg["success_rate"] = float(np.mean([r["success"] for r in results]))
        agg["resection_success_rate"] = float(
            np.mean([r["resection_success"] for r in results]))
        agg["grasp_lift_success_rate"] = float(
            np.mean([r["grasp_lift_success"] for r in results]))
        # per-stage reach rates: a policy that never gets the cutter onto the
        # tissue and one that gets there and fails to close read the same in
        # success_rate and completely differently here
        for s in Episode.STAGES:
            agg[f"stage_{s}_rate"] = float(
                np.mean([r[f"stage_{s}"] for r in results]))
        agg["n_episodes"] = len(results)
    (out / "summary.json").write_text(json.dumps(
        dict(config=vars(args), episodes=results, aggregate=agg),
        indent=2, default=float))
    print(f"\nsuccess rate {agg.get('success_rate', 0.0)*100:.0f}% (all "
          f"{len(Episode.STAGES)} stages) over {len(results)} episode(s)")
    for s in Episode.STAGES:
        print(f"  {s:<14s} {agg.get(f'stage_{s}_rate', 0.0)*100:3.0f}%")
    print(f"total wall {time.time()-t_run:.0f} s -> {out/'summary.json'}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
