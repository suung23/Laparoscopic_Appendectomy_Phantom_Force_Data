"""Create a local Appendix300-derived UV appearance candidate.

The image source remains a single reviewed frame and hand-selected tissue ROI;
this is an appearance transfer experiment, not segmentation, albedo recovery,
3-D registration, or clinical validation. Texture extraction uses regular
Python; pass --source to additionally author a USD material layer in Isaac Sim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageFilter
DATASET = "Appendix300"
ATTRIBUTION = "Appendix300, DOI 10.25532/OPARA-1173, CC BY 4.0"
# Reviewed against Center1_t12.png (1920 x 1080); boxes avoid the visible tool.
ROIS = {
    "appendix": (900, 240, 1160, 480),
    "mesoappendix": (260, 690, 610, 960),
}
TARGETS = {
    "appendix": "/World/Atlas/appendix/render",
    "mesoappendix": "/World/Atlas/mesoappendix/render",
}


def quilt_patch_texture(frame, box, output, seed=300, size=1024, patch_size=96, overlap=24):
    """Synthesize a non-mirrored texture field from overlapping ROI crops.

    This reduces the obvious mirror symmetry of the initial proof of concept.
    It is stochastic appearance synthesis, not surface registration or albedo
    recovery. A fixed seed makes the candidate reproducible.
    """
    patch = frame.crop(box).convert("RGB")
    source_patch_size = list(patch.size)
    # Conservatively attenuate broad lamp falloff and replace near-white
    # specular cores locally. This cannot recover true intrinsic albedo.
    rgb = np.asarray(patch, dtype=np.float32)
    low = np.asarray(patch.convert("L").filter(ImageFilter.GaussianBlur(28)), dtype=np.float32)
    target = float(np.median(low))
    scale = np.sqrt(target / np.maximum(low, 1.))
    patch = Image.fromarray(np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8), "RGB")
    hsv = np.asarray(patch.convert("HSV"))
    # Suppress some compact, near-neutral lamp glints from the base-color
    # candidate. Residual highlights remain; this is not intrinsic-albedo
    # estimation or a calibrated specular/diffuse decomposition.
    mask = Image.fromarray(np.uint8((hsv[..., 2] > 198) & (hsv[..., 1] < 145)) * 255)
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5))
    patch = Image.composite(patch.filter(ImageFilter.MedianFilter(9)), patch, mask)
    source = np.asarray(patch, dtype=np.float32)
    h, w = source.shape[:2]
    if patch_size > min(h, w) or overlap <= 0 or overlap >= patch_size:
        raise ValueError("Quilting patch/overlap dimensions do not fit the ROI")
    if size <= patch_size:
        raise ValueError("Texture size must be larger than a quilting patch")

    rng = np.random.default_rng(seed)
    stride = patch_size - overlap
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    valid = np.zeros((size, size), dtype=bool)
    # Each grid cell chooses a low-error crop from a seeded candidate pool.
    candidates_per_cell = 32
    for y in range(0, size, stride):
        for x in range(0, size, stride):
            ph, pw = min(patch_size, size - y), min(patch_size, size - x)
            xs = rng.integers(0, w - pw + 1, size=candidates_per_cell)
            ys = rng.integers(0, h - ph + 1, size=candidates_per_cell)
            crops = [source[yy:yy+ph, xx:xx+pw] for xx, yy in zip(xs, ys)]
            left = x > 0
            top = y > 0
            ol = min(overlap, pw) if left else 0
            ot = min(overlap, ph) if top else 0
            scores = []
            for candidate in crops:
                errors = []
                if ol:
                    errors.append(np.mean((candidate[:, :ol] - canvas[y:y+ph, x:x+ol]) ** 2))
                if ot:
                    errors.append(np.mean((candidate[:ot] - canvas[y:y+ot, x:x+pw]) ** 2))
                scores.append(float(np.mean(errors)) if errors else 0.)
            # Randomly select among the closest candidates to avoid a fixed
            # low-variance texture while keeping seams reasonably compatible.
            best = np.argsort(scores)[:min(5, len(scores))]
            chosen = crops[int(rng.choice(best))]
            alpha = np.ones((ph, pw), dtype=np.float32)
            if ol:
                ramp = np.linspace(0., 1., ol, endpoint=True, dtype=np.float32)
                alpha[:, :ol] = np.minimum(alpha[:, :ol], ramp[None, :])
            if ot:
                ramp = np.linspace(0., 1., ot, endpoint=True, dtype=np.float32)
                alpha[:ot, :] = np.minimum(alpha[:ot, :], ramp[:, None])
            region = canvas[y:y+ph, x:x+pw]
            was_valid = valid[y:y+ph, x:x+pw]
            blend = alpha[..., None]
            region[:] = np.where(was_valid[..., None],
                                 region * (1. - blend) + chosen * blend,
                                 chosen)
            valid[y:y+ph, x:x+pw] = True

    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB").save(output)
    return {"source_patch_px": source_patch_size, "atlas_texture_px": [size, size],
            "synthesis": "seeded_overlapping_patch_quilting",
            "seed": seed, "quilting_patch_px": patch_size,
            "overlap_px": overlap, "candidate_crops_per_cell": candidates_per_cell}


def build(reference, out, ffmpeg, sources=None):
    reference, out = map(lambda p: Path(p).resolve(), (reference, out))
    sources = [Path(p).resolve() for p in sources] if sources else None
    out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((reference / "provenance.json").read_text(encoding="utf-8"))
    item = next(x for x in provenance["files"] if x["local_name"] == "Center1_014.mp4")
    video = reference / item["local_name"]
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    if video_hash != item["sha256"]:
        raise ValueError("Appendix300 video hash does not match the local provenance record")
    frame_path = out / "center1_t12.png"
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-ss", "12",
                    "-i", str(video), "-frames:v", "1", "-y", str(frame_path)], check=True)
    frame = Image.open(frame_path).convert("RGB")
    if frame.size != (1920, 1080):
        raise ValueError(f"Expected the manually reviewed 1920x1080 source frame, got {frame.size}")

    tiles = {}
    for name, box in ROIS.items():
        texture_path = out / f"{name}_appendix300_rgb.png"
        tiles[name] = {"roi_xyxy": list(box), "dimensions":
                       quilt_patch_texture(frame, box, texture_path, seed=300 + len(tiles)),
                       "texture_sha256": hashlib.sha256(texture_path.read_bytes()).hexdigest()}

    report = {
        "status": "texture_sources_ready_usd_pending" if sources is None else "unreviewed_video_derived_appearance_candidate",
        "dataset": DATASET, "attribution": ATTRIBUTION,
        "license": provenance["license"], "source_doi": provenance["doi"],
        "source_video": item["name"], "source_video_sha256": video_hash,
        "source_frame_time_seconds": 12, "source_frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "rois_and_textures": tiles,
        "geometry_unchanged": True,
        "physics_scope": "Static USD topology/material overlay only; runtime tissue dynamics were not exercised",
        "measured_albedo": False, "measured_brdf": False,
        "clinical_visual_validation": False,
        "limitations": [
            "ROI tissue was selected visually and was not instance-segmented or verified as the named anatomy",
            "Observed lighting, white balance, specular highlights, smoke and blood may remain baked into RGB",
            "Seeded patch quilting synthesizes a non-mirrored texture field but repeats source appearance statistics and is not camera-to-mesh registration",
            "This transfer does not add recovered 3-D anatomy or tissue mechanics",
        ],
    }
    if sources is not None:
        from pxr import Usd
        from reference_appearance import author_texture
        stage = Usd.Stage.CreateNew(str(out / "appearance.usda"))
        for source in sources:
            stage.GetRootLayer().subLayerPaths.append(
                os.path.relpath(source, out).replace("\\", "/"))
        records = []
        for name, prim in TARGETS.items():
            if not stage.GetPrimAtPath(prim):
                raise ValueError(f"Target mesh is absent from source scene: {prim}")
            records.append(author_texture(stage, prim, f"{name}_appendix300_rgb.png",
                repeat=1., roughness=.34, material_path=f"/World/Appendix300VideoLook/{name}"))
        stage.GetRootLayer().Save()
        report["mapped_meshes"] = records
        report["source_scene_layers"] = [{
            "file": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        } for source in sources]
        report["authored_stage_sha256"] = hashlib.sha256(
            (out / "appearance.usda").read_bytes()).hexdigest()
    (out / "appearance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Ignored local Appendix300 reference directory")
    parser.add_argument("--source", type=Path, nargs="+",
                        help="USD scene layers, in composition order; omit to prepare textures only")
    parser.add_argument("--out", type=Path, required=True, help="Local candidate output directory")
    parser.add_argument("--ffmpeg", type=Path, default=Path(".cache/bin/ffmpeg.exe"))
    args = parser.parse_args()
    build(args.reference, args.out, args.ffmpeg, args.source)
