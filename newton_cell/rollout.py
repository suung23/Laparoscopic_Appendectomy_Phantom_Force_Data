#!/usr/bin/env python3
"""Run `isaac/vla_rollout.py` against the Newton backend, unmodified.

    .venv-newton/bin/python newton_cell/rollout.py --episodes 1 --seconds 20 --video

`isaac/vla_rollout.py` -- the episode recorder, the scorer, the npz writer and
the ffmpeg mp4 step -- contains no Isaac imports at all, and it resolves the
environment with a LATE `from vla_env import LaparoVLAEnv` inside `main()`. That
one detail is what lets both backends share it: this shim imports the module
(which puts `isaac/` on `sys.path`), then puts `newton_cell/` AHEAD of it, so
the late import binds to the Newton `vla_env` instead.

Nothing in `isaac/` is edited. Running `isaac/vla_rollout.py` directly still
gets the Isaac backend exactly as before; the backend is chosen by which entry
point you launch, so the default behaviour of the RTX workstation is untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(ROOT / "isaac"))
import vla_rollout                                   # noqa: E402

# vla_rollout put isaac/ at sys.path[0] on import; go in front of it so the
# late `from vla_env import ...` resolves here.
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--gui" in argv:
        raise SystemExit(
            "--gui is not available on the Newton backend: this host has no RT "
            "cores and no display, which is why the backend exists. Videos come "
            "out of --video exactly the same way.")
    import vla_env
    if Path(vla_env.__file__).parent != HERE:
        raise SystemExit(f"wrong vla_env bound: {vla_env.__file__}")
    print(f"backend: newton ({vla_env.__file__})", flush=True)

    out = _out_dir(argv)
    lock = _claim(out)
    want_video = "--video" in argv
    front_dir = out / "frames" / "front"
    if want_video:
        import shutil
        if front_dir.exists():
            shutil.rmtree(front_dir)
        vla_env.FRONT_OUT = front_dir
        vla_env.FRONT_EVERY = _int_opt(argv, "--image-every", 1)

    try:
        rc = vla_rollout.main(argv)
        if want_video:
            _encode_front(front_dir, out, _int_opt(argv, "--control-hz", 30))
    finally:
        lock.unlink(missing_ok=True)
    return rc


def _claim(out: Path) -> Path:
    """Refuse to share an output directory with another live rollout.

    Two rollouts writing the same `frames/ep000/` overwrite each other's PNGs
    frame by frame, and whichever finishes last encodes the MIXTURE. That is
    not hypothetical: it produced a video whose floor flickered between the
    blue surgical drape and bare grey, because the two runs were built from
    different revisions of the scene. Nothing in the output looked corrupt --
    each frame was individually valid -- which is exactly why it needs a guard
    rather than a code review.
    """
    import os
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".rollout.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
        except Exception:
            pid = -1
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise SystemExit(
                f"another rollout (pid {pid}) is already writing to "
                f"{out}.\nTwo runs sharing one output directory interleave "
                f"their frames and the video comes out a mixture of both. "
                f"Use a different --out, or wait for pid {pid} to finish.")
        print(f"    (clearing stale lock from dead pid {pid})", flush=True)
    lock.write_text(f"{os.getpid()}\n")
    return lock


def _out_dir(argv) -> Path:
    o = "out/vla_eval"
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            o = argv[i + 1]
        elif a.startswith("--out="):
            o = a.split("=", 1)[1]
    p = Path(o)
    return p if p.is_absolute() else ROOT / p


def _int_opt(argv, name, default) -> int:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return int(float(argv[i + 1]))
        if a.startswith(name + "="):
            return int(float(a.split("=", 1)[1]))
    return default


def _encode_front(frames: Path, out: Path, fps: int) -> None:
    """H.264 the front frames, the same way vla_rollout does its own."""
    import subprocess
    pngs = sorted(frames.glob("front_*.png")) if frames.is_dir() else []
    if not pngs:
        print("    (no front frames to encode)", flush=True)
        return
    dest = out / "episode_000_front.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", str(frames / "front_%05d.png"), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "18", str(dest)]
    try:
        subprocess.run(cmd, check=True)
        print(f"    {dest.relative_to(ROOT)}", flush=True)
    except Exception as e:
        print(f"    front mp4 FAILED: {e}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
