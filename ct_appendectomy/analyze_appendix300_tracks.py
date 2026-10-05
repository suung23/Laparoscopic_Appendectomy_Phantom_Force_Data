"""Measure whether local Appendix300 clips support rigid multi-view geometry.

This is a feasibility probe, not a reconstruction. Endoscopic tissue moves,
the camera is uncalibrated, and monocular translation has unknown scale.
Only aggregate feature-track statistics and video hashes are written.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _frames(video, ffmpeg, interval_s, max_dimension, work):
    pattern = work / "frame_%05d.png"
    rate = f"1/{interval_s:g}"
    filt = f"scale={max_dimension}:{max_dimension}:force_original_aspect_ratio=decrease:force_divisible_by=2,fps={rate}"
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(video),
               "-vf", filt, "-fps_mode", "vfr", str(pattern)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FFmpeg could not decode {video.name}: {result.stderr[-500:]}")
    return sorted(work.glob("frame_*.png"))


def _pair_metrics(first, second, matcher, focal_ratio):
    kp0, desc0, gray0, time0 = first
    kp1, desc1, gray1, time1 = second
    h, w = gray0.shape
    if desc0 is None or desc1 is None:
        return {"time_s": [time0, time1], "matches": 0, "essential_inliers": 0}
    candidates = matcher.knnMatch(desc0, desc1, k=2)
    good = [a for pair in candidates if len(pair) == 2 for a, b in [pair] if a.distance < .72*b.distance]
    record = {"time_s": [time0, time1], "keypoints": [len(kp0), len(kp1)],
              "matches": len(good), "essential_inliers": 0}
    if len(good) < 12:
        return record
    p0 = np.float64([kp0[m.queryIdx].pt for m in good])
    p1 = np.float64([kp1[m.trainIdx].pt for m in good])
    focal = focal_ratio * max(w, h)
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
    E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC,
                                   prob=.999, threshold=1.5)
    if E is None or mask is None:
        return record
    _, R, translation, pose_mask = cv2.recoverPose(E, p0, p1, K, mask=mask)
    keep = pose_mask.ravel().astype(bool)
    record["essential_inliers"] = int(keep.sum())
    record["inlier_fraction"] = float(keep.mean())
    if keep.any():
        points = p0[keep]
        cols = np.clip((points[:, 0] / w * 8).astype(int), 0, 7)
        rows = np.clip((points[:, 1] / h * 6).astype(int), 0, 5)
        record["inlier_grid_cells_8x6"] = int(len(set(zip(cols.tolist(), rows.tolist()))))
        rays = np.c_[(p0[keep, 0]-w/2)/focal, (p0[keep, 1]-h/2)/focal,
                     np.ones(int(keep.sum()))]
        rotated = rays @ R.T
        projected = np.c_[focal*rotated[:, 0]/rotated[:, 2]+w/2,
                          focal*rotated[:, 1]/rotated[:, 2]+h/2]
        residual = np.linalg.norm(p1[keep]-projected, axis=1)
        record["rotation_compensated_residual_px_p50_p90"] = np.quantile(residual, [.5, .9]).tolist()
    record["translation_direction"] = "estimated up to scale; not serialized"
    return record


def analyze(video, ffmpeg, interval_s=.5, max_dimension=960, focal_ratio=.9):
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    with tempfile.TemporaryDirectory(prefix="appendix300_tracks_") as temp:
        frame_paths = _frames(video, Path(ffmpeg).resolve(), interval_s, max_dimension, Path(temp))
        sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.025)
        frames = []
        for i, path in enumerate(frame_paths):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            keypoints, descriptors = sift.detectAndCompute(image, None)
            frames.append((keypoints, descriptors, image, i*interval_s))
        matcher = cv2.BFMatcher()
        pairs = [_pair_metrics(a, b, matcher, focal_ratio) for a, b in zip(frames, frames[1:])]
    inliers = np.asarray([p["essential_inliers"] for p in pairs], dtype=np.float64)
    coverage = np.asarray([p.get("inlier_grid_cells_8x6", 0) for p in pairs], dtype=np.float64)
    return {
        "video": video.name,
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "sample_interval_s": interval_s,
        "sampled_frames": len(frames),
        "median_keypoints": int(np.median([len(f[0]) for f in frames])) if frames else 0,
        "adjacent_pairs": len(pairs),
        "median_essential_inliers": float(np.median(inliers)) if len(inliers) else 0,
        "pairs_with_at_least_20_inliers": int((inliers >= 20).sum()),
        "median_inlier_grid_cells_8x6": float(np.median(coverage)) if len(coverage) else 0,
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=.5)
    parser.add_argument("--max-dimension", type=int, default=960)
    parser.add_argument("--focal-ratio", type=float, default=.9,
                        help="Uncalibrated focal length / maximum image dimension")
    args = parser.parse_args()
    if args.interval <= 0 or args.max_dimension < 128 or not 0 < args.focal_ratio < 4:
        parser.error("Invalid sampling or camera assumption")
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    expected = {item["local_name"]: item["sha256"] for item in provenance["files"]}
    results = []
    for video in args.video:
        video = video.resolve()
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        if video.name not in expected or expected[video.name] != digest:
            raise ValueError(f"Video is absent from or differs from provenance: {video.name}")
        results.append(analyze(video, args.ffmpeg, args.interval,
                                args.max_dimension, args.focal_ratio))
    report = {
        "status": "exploratory feature-track feasibility; no 3D reconstruction",
        "dataset": provenance["dataset"],
        "doi": provenance["doi"],
        "license": provenance["license"],
        "sampling": {"interval_s": args.interval, "max_dimension_px": args.max_dimension,
                     "feature": "SIFT", "matching": "Lowe ratio 0.72",
                     "relative_pose": "RANSAC essential matrix; focal ratio assumed, uncalibrated"},
        "limitations": [
            "Endoscope calibration, motion, and depth scale are unavailable",
            "Tissue deformation, occlusion, specular highlights, and tools violate rigid SfM",
            "Feature tracks are not anatomical labels or a validated surface reconstruction",
            "No patient metadata or image pixels are copied into the report",
        ],
        "clips": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "clips": [
        {k: c[k] for k in ("video", "sampled_frames", "adjacent_pairs",
                            "median_essential_inliers", "pairs_with_at_least_20_inliers",
                            "median_inlier_grid_cells_8x6")} for c in results]}, indent=2))


if __name__ == "__main__":
    main()
