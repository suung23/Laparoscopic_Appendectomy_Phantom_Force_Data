"""Local GPU baseline; does not validate surgical or clinical fidelity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newton_cell"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))

import numpy as np
import warp as wp
from newton.solvers import SolverVBD
from scene import CellScene, add_args


def volumes(q, indices):
    p = q[indices]
    return np.einsum("ij,ij->i", np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), p[:, 3]-p[:, 0]) / 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()
    if args.frames < 120:
        ap.error("baseline requires at least 120 frames (2 seconds)")
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    mismatches = []
    for f in manifest["files"]:
        p = ROOT / f["path"]
        b = p.read_bytes() if p.is_file() else b""
        if hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest() != f["sha"]:
            mismatches.append(f["path"])
    if mismatches:
        raise RuntimeError(f"Upstream snapshot modified or incomplete: {mismatches}")

    wp.init()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("CUDA required; no silent CPU fallback")
    cfg = add_args(argparse.ArgumentParser()).parse_args(["--device", "cuda:0", "--no-phantom"])
    with wp.ScopedDevice(device):
        scene = CellScene(cfg)
        model = scene.build()
        solver = SolverVBD(model=model, iterations=cfg.iterations, particle_enable_self_contact=False)
        s0, s1 = model.state(), model.state()
        control, contacts = model.control(), model.contacts()
        q0 = s0.particle_q.numpy().copy()
        indices = model.tet_indices.numpy().reshape(-1, 4)
        v0 = volumes(q0, indices)
        pinned = np.concatenate([x for x in scene.pinned.values() if len(x)])
        finite = True
        frames = 0
        start = time.perf_counter()
        for frame in range(args.frames):
            for _ in range(cfg.substeps):
                s0.clear_forces()
                model.collide(s0, contacts)
                solver.step(s0, s1, control, contacts, 1 / cfg.fps / cfg.substeps)
                s0, s1 = s1, s0
            q = s0.particle_q.numpy()
            finite = bool(np.isfinite(q).all() and np.isfinite(s0.particle_qd.numpy()).all())
            frames = frame + 1
            if not finite:
                break
        wp.synchronize()
        wall = time.perf_counter() - start
        q = s0.particle_q.numpy()
        v = volumes(q, indices)
        drift = float(np.linalg.norm(q[pinned] - q0[pinned], axis=1).max())
        speed = float(np.linalg.norm(s0.particle_qd.numpy(), axis=1).max())
        inverted = int(np.count_nonzero(v * v0 < 0))
        organs = {}
        for name, (i, j) in scene.organ_span.items():
            dx = np.linalg.norm(q[i:j] - q0[i:j], axis=1) * 1000
            organs[name] = {"mean_displacement_mm": float(dx.mean()), "max_displacement_mm": float(dx.max())}
        passed = finite and frames == args.frames and drift < 1e-6 and speed < .05 and inverted == 0
        report = {
            "scope": "tissue-only rest stability; no instruments, rendering, cutting, or clinical validation",
            "upstream_commit": manifest["commit"], "upstream_hashes_verified": len(manifest["files"]),
            "device": device.name, "python": sys.version.split()[0],
            "versions": {n: importlib.metadata.version(n) for n in ("newton", "warp-lang", "numpy", "trimesh", "scipy")},
            "particles": model.particle_count, "tetrahedra": model.tet_count,
            "solver": "SolverVBD", "iterations": cfg.iterations, "substeps": cfg.substeps,
            "self_contact": False, "mass_floor_ratio": cfg.mass_floor,
            "frames": frames, "sim_seconds": frames / cfg.fps, "wall_seconds_including_kernel_compile": wall,
            "finite": finite, "pinned_max_drift_mm": drift * 1000, "final_max_speed_m_s": speed,
            "orientation_flipped_tetrahedra": inverted, "organs": organs,
            "criteria": {"finite": True, "pin_drift_below_mm": .001, "final_speed_below_m_s": .05, "orientation_flips": 0},
            "passed": passed,
        }
        out = ROOT / "validation"
        out.mkdir(exist_ok=True)
        (out / "tissue-baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        np.savez_compressed(out / "tissue-baseline.npz", initial=q0, final=q, indices=indices)
        print(json.dumps(report, indent=2))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
