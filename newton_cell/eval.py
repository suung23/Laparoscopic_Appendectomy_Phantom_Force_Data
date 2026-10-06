#!/usr/bin/env python3
"""Evaluation entry point for the H100 backend, with render presets.

    newton_cell/eval.py --policy ckpt.py:Policy --episodes 4 --preset fast

The defaults elsewhere in this package are tuned for a presentation video, and
running those inside a training loop wastes hours. This wraps `rollout.py` (and
through it, `isaac/vla_rollout.py` unmodified) with a small set of presets that
trade image quality for wall clock, plus the individual knobs if you want them.

Everything not recognised here is passed straight through, so every option
`isaac/vla_rollout.py` documents still works.

Why presets rather than "just use good defaults": there is no single right
answer. Rendering is the whole cost on this host -- an H100 has no RT cores, so
BVH traversal runs in software -- and what you are willing to pay for it depends
entirely on whether a human is going to look at the result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# name -> env overrides. `front` is the big one: the overview camera renders a
# scene five times the size of the clinical one and costs ~85 % of a showcase
# run, for a view no policy ever sees.
PRESETS = {
    "metrics": dict(front=False, samples=1, bounces=0,
                    _help="no usable imagery, scores only -- fastest"),
    "fast": dict(front=False, samples=6, bounces=0,
                 _help="endoscope + D405, watchable; the training-loop default"),
    "balanced": dict(front=False, samples=12, bounces=1,
                     _help="indirect light back on; for checkpoint reviews"),
    "showcase": dict(front=True, samples=16, bounces=1, front_samples=6,
                     _help="all three cameras at video quality -- slow"),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="balanced")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--bounces", type=int)
    ap.add_argument("--front-samples", type=int)
    ap.add_argument("--front", dest="front", action="store_true", default=None)
    ap.add_argument("--no-front", dest="front", action="store_false")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--physics-only", action="store_true",
                    help="skip path tracing and return physics/task state only; "
                         "not valid for RGB policy evaluation")
    mine, rest = ap.parse_known_args(argv)

    if mine.list_presets:
        w = max(len(k) for k in PRESETS)
        for name, cfg in PRESETS.items():
            knobs = ", ".join(f"{k}={v}" for k, v in cfg.items()
                              if not k.startswith("_"))
            print(f"  {name:<{w}}  {cfg['_help']}\n  {'':<{w}}  {knobs}")
        return 0

    over = {k: v for k, v in PRESETS[mine.preset].items()
            if not k.startswith("_")}
    for k in ("samples", "bounces", "front_samples", "front"):
        v = getattr(mine, k)
        if v is not None:
            over[k] = v

    import vla_env
    vla_env.OVERRIDES.update(over)
    vla_env.PHYSICS_ONLY = mine.physics_only
    if mine.physics_only:
        print("physics-only observations: RGB/depth rendering disabled; "
              "do not treat this as a visual-policy rollout", flush=True)
    print(f"preset {mine.preset}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(over.items())), flush=True)

    import rollout
    return rollout.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
