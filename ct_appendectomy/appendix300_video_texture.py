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


def seamless_mirror_patch(frame, box, output):
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
    mask = Image.fromarray(np.uint8((hsv[..., 2] > 226) & (hsv[..., 1] < 105)) * 255)
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5))
    patch = Image.composite(patch.filter(ImageFilter.MedianFilter(9)), patch, mask)
    # Mirror tiling closes the UV boundary; retain texture detail near the
    # source scale by repeating the seamless tile within the atlas.
    tile = Image.new("RGB", (patch.width * 2, patch.height * 2))
    tile.paste(patch, (0, 0))
    tile.paste(patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT), (patch.width, 0))
    tile.paste(patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM), (0, patch.height))
    tile.paste(patch.transpose(Image.Transpose.ROTATE_180), (patch.width, patch.height))
    tile = tile.resize((256, 256), Image.Resampling.LANCZOS)
    tile.resize((1024, 1024), Image.Resampling.NEAREST).save(output)
    return {"source_patch_px": source_patch_size, "periodic_tile_px": list(tile.size),
            "atlas_texture_px": [1024, 1024]}


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
                       seamless_mirror_patch(frame, box, texture_path),
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
            "Mirror tiling and UV atlas mapping can repeat image features; no calibrated camera-to-mesh registration",
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
