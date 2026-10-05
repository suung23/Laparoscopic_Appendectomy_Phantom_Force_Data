"""Build a local, provenance-checked Appendix300 multi-frame texture candidate.

Adjacent laparoscopic frames are translation-aligned within each manually
reviewed ROI and combined with a per-pixel median to reduce transient glints.
This is not non-rigid 3-D registration, tissue segmentation, or albedo recovery.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image

from appendix300_video_texture import ATTRIBUTION, DATASET, ROIS, quilt_patch_image


def _gray(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114


def phase_translation(reference, moving, max_shift=24):
    """Estimate integer translation to sample `moving` into reference space."""
    ref, mov = _gray(reference), _gray(moving)
    if ref.shape != mov.shape or min(ref.shape) < 32:
        raise ValueError("Phase-correlation inputs must have matching dimensions >= 32px")
    if np.array_equal(ref, mov):
        return 0, 0, 1.0
    h, w = ref.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    ref = (ref - ref.mean()) * window
    mov = (mov - mov.mean()) * window
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cross /= np.maximum(np.abs(cross), 1e-9)
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy = int(peak_y if peak_y <= h // 2 else peak_y - h)
    dx = int(peak_x if peak_x <= w // 2 else peak_x - w)
    # With ref * conj(moving), the inverse-DFT peak is the source-space
    # sampling offset used to read moving into reference coordinates.
    sample_y, sample_x = dy, dx
    if abs(sample_y) > max_shift or abs(sample_x) > max_shift:
        raise ValueError(f"Estimated alignment exceeds {max_shift}px: {(sample_x, sample_y)}")
    suppressed = correlation.copy()
    yy = (peak_y + np.arange(-3, 4)) % h
    xx = (peak_x + np.arange(-3, 4)) % w
    suppressed[np.ix_(yy, xx)] = -np.inf
    second = max(float(np.max(suppressed)), 1e-9)
    peak_ratio = float(correlation[peak_y, peak_x] / second)
    return sample_x, sample_y, peak_ratio


def translate(image, dx, dy):
    """Translate without wraparound; exposed pixels are edge-padded."""
    source = np.asarray(image.convert("RGB"))
    h, w = source.shape[:2]
    padded = np.pad(source, ((abs(dy), abs(dy)), (abs(dx), abs(dx)), (0, 0)), mode="edge")
    y0, x0 = abs(dy) - dy, abs(dx) - dx
    return Image.fromarray(padded[y0:y0 + h, x0:x0 + w])


def rebind_reviewed_uv_layer(parent_layer, output_dir):
    """Reuse a provenance-checked static USD UV layer with new local textures.

    The parent was already authored against the CT/fat scene. This operation
    changes only the two RGB asset paths; it does not regenerate or alter UVs,
    topology, simulation attributes, or the referenced scene layers.
    """
    parent_layer = Path(parent_layer).resolve()
    output_dir = Path(output_dir).resolve()
    parent_report_path = parent_layer.with_name("appearance.json")
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    parent_hash = hashlib.sha256(parent_layer.read_bytes()).hexdigest()
    if parent_report.get("dataset") != DATASET or parent_report.get("license") != "CC BY 4.0":
        raise ValueError("Parent USD layer is not linked to licensed Appendix300 provenance")
    if not parent_report.get("geometry_unchanged") or parent_report.get(
            "authored_stage_sha256") != parent_hash:
        raise ValueError("Parent USD layer hash/geometry claim does not match its report")
    expected_meshes = {"/World/Atlas/appendix/render", "/World/Atlas/mesoappendix/render"}
    actual_meshes = {record.get("mesh") for record in parent_report.get("mapped_meshes", [])}
    if actual_meshes != expected_meshes:
        raise ValueError("Parent USD layer does not map exactly the expected tissue meshes")
    text = parent_layer.read_text(encoding="utf-8")
    substitutions = {
        "appendix_appendix300_rgb.png": "appendix_appendix300_temporal_rgb.png",
        "mesoappendix_appendix300_rgb.png": "mesoappendix_appendix300_temporal_rgb.png",
    }
    for old_name, new_name in substitutions.items():
        old_asset, new_asset = f"@{old_name}@", f"@{new_name}@"
        if text.count(old_asset) != 1:
            raise ValueError(f"Expected exactly one reviewed USD texture reference: {old_name}")
        if not (output_dir / new_name).is_file():
            raise ValueError(f"Temporal texture is missing from output directory: {new_name}")
        text = text.replace(old_asset, new_asset)
    sublayer_block = re.search(r"subLayers\s*=\s*\[(.*?)\]", text, flags=re.S)
    if sublayer_block is None:
        raise ValueError("Parent USD layer has no explicit sublayers")
    sublayers = re.findall(r"@([^@]+)@", sublayer_block.group(1))
    resolved_sources = [(item, (output_dir / item).resolve()) for item in sublayers]
    unresolved = [item for item, path in resolved_sources if not path.is_file()]
    if unresolved:
        raise ValueError(f"Parent USD sublayers do not resolve from output directory: {unresolved}")
    source_records = parent_report.get("source_scene_layers", [])
    if len(source_records) != len(resolved_sources):
        raise ValueError("Parent report does not hash each referenced scene sublayer")
    verified_sources = []
    for (relative, path), record in zip(resolved_sources, source_records):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise ValueError(f"Referenced scene layer changed since parent review: {relative}")
        verified_sources.append({"path": relative, "sha256": digest})
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "appearance.usda"
    target.write_text(text, encoding="utf-8")
    return {
        "path": target.name,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "parent_layer_sha256": parent_hash,
        "referenced_sublayers": sublayers,
        "source_scene_layers": verified_sources,
        "changed_properties": ["Appendix RGB texture asset", "Mesoappendix RGB texture asset"],
        "uvs_reused_unchanged": True,
        "geometry_unchanged_by_layer_operation": True,
        "physics_unchanged_by_layer_operation": True,
        "usd_runtime_load_verified": False,
    }


def extract_frames(video, ffmpeg, output, times):
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, timestamp in enumerate(times):
        path = output / f"frame_{index:02d}_{timestamp:.3f}s.png"
        result = subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "info", "-ss",
                        f"{timestamp:.6f}", "-i", str(video), "-frames:v", "1", "-y",
                        "-vf", "showinfo", str(path)], check=True, capture_output=True, text=True)
        pts = re.search(r"pts_time:([0-9.]+)", result.stderr)
        if pts is None:
            raise ValueError(f"Could not read decoded frame PTS at requested {timestamp}s")
        # Input-side seeking resets the decoded stream timestamp near zero.
        actual_timestamp = timestamp + float(pts.group(1))
        with Image.open(path) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"Unexpected frame dimensions at {timestamp}s: {image.size}")
        frames.append((actual_timestamp, path))
    return frames


def build(reference_dir, out_dir, ffmpeg, center_seconds=12., frame_count=9,
          spacing_seconds=.06, sources=None, parent_layer=None):
    reference_dir, out_dir = Path(reference_dir).resolve(), Path(out_dir).resolve()
    ffmpeg = Path(ffmpeg).resolve()
    provenance_path = reference_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("doi") != "10.25532/OPARA-1173" or provenance.get("license") != "CC BY 4.0":
        raise ValueError("Local source manifest is not the licensed Appendix300 record")
    item = next((f for f in provenance["files"] if f["local_name"] == "Center1_014.mp4"), None)
    if item is None:
        raise ValueError("Source manifest does not include the expected local snippet")
    video = reference_dir / item["local_name"]
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    if video_hash != item["sha256"]:
        raise ValueError("Appendix300 source video hash does not match its local manifest")
    if frame_count < 3 or frame_count % 2 == 0 or frame_count > 21:
        raise ValueError("frame_count must be odd and between 3 and 21")
    if not np.isfinite(spacing_seconds) or spacing_seconds <= 0:
        raise ValueError("spacing_seconds must be positive and finite")
    if center_seconds - ((frame_count // 2) * spacing_seconds) < 0:
        raise ValueError("The requested time window starts before the source clip")
    times = [center_seconds + i * spacing_seconds for i in range(-(frame_count // 2),
                                                                    frame_count // 2 + 1)]
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_records = extract_frames(video, ffmpeg, out_dir / "frames", times)
    full_frames = []
    for timestamp, path in frame_records:
        with Image.open(path) as im:
            full_frames.append((timestamp, im.convert("RGB")))

    rois, alignment_report = {}, {}
    for index, (name, box) in enumerate(ROIS.items()):
        crops = [(timestamp, image.crop(box)) for timestamp, image in full_frames]
        reference = crops[len(crops) // 2][1]
        aligned, transforms, shifts = [], [], []
        for timestamp, crop in crops:
            dx, dy, ratio = phase_translation(reference, crop)
            aligned_crop = translate(crop, dx, dy)
            aligned.append(np.asarray(aligned_crop, dtype=np.uint8))
            shifts.append((dx, dy))
            transforms.append({"time_seconds": timestamp, "sample_shift_xy_px": [dx, dy],
                               "peak_to_sidelobe_ratio": round(ratio, 5)})
        temporal = np.median(np.stack(aligned, axis=0), axis=0).astype(np.uint8)
        # Exclude replicated border pixels so only source-observed texels enter
        # the temporal median and downstream quilting.
        pad_x = max(abs(dx) for dx, _ in shifts)
        pad_y = max(abs(dy) for _, dy in shifts)
        if pad_x * 2 >= temporal.shape[1] or pad_y * 2 >= temporal.shape[0]:
            raise ValueError("Alignment leaves no common observed area in the ROI")
        temporal = temporal[pad_y:temporal.shape[0] - pad_y if pad_y else None,
                            pad_x:temporal.shape[1] - pad_x if pad_x else None]
        temporal_path = out_dir / f"{name}_temporal_median.png"
        Image.fromarray(temporal).save(temporal_path)
        texture_path = out_dir / f"{name}_appendix300_temporal_rgb.png"
        # Keep an un-repeated crop as the primary candidate. Its UV scale is
        # still approximate, but this avoids fabricating a tiled surface field.
        source_patch = Image.fromarray(temporal)
        direct_texture = source_patch.resize((1024, 1024), Image.Resampling.LANCZOS)
        direct_texture.save(texture_path)
        quilt_path = out_dir / f"{name}_appendix300_temporal_quilt_rgb.png"
        quilt_dimensions = quilt_patch_image(source_patch, quilt_path,
                                             seed=3300 + index, size=1024,
                                             patch_size=128, overlap=32)
        rois[name] = {"roi_xyxy": list(box), "valid_border_removed_px": [pad_x, pad_y],
                      "median_patch_sha256": hashlib.sha256(
            temporal_path.read_bytes()).hexdigest(), "texture_sha256": hashlib.sha256(
            texture_path.read_bytes()).hexdigest(),
            "texture_method": "single_temporal_patch_resampled_without_tiling",
            "median_patch_dimensions_px": [int(temporal.shape[1]), int(temporal.shape[0])],
            "texture_dimensions_px": [1024, 1024],
            "quilted_alternative_sha256": hashlib.sha256(quilt_path.read_bytes()).hexdigest(),
            "quilted_alternative_dimensions": quilt_dimensions}
        alignment_report[name] = transforms

    report = {
        "status": "multi_frame_texture_candidate_usd_pending" if not sources else
                  "multi_frame_usd_candidate_unrendered",
        "dataset": DATASET, "attribution": ATTRIBUTION,
        "license": provenance["license"], "source_doi": provenance["doi"],
        "source_video": item["name"], "source_video_sha256": video_hash,
        "requested_frame_times_seconds": times,
        "source_frames": [{"decoded_time_seconds": t, "sha256": hashlib.sha256(
            p.read_bytes()).hexdigest()} for t, p in frame_records],
        "alignment_method": "per-ROI integer phase correlation to middle frame",
        "alignment_by_roi": alignment_report, "rois_and_textures": rois,
        "texture_mapping": "Each single observed crop fills one UV map; no 3-D camera projection or patient correspondence",
        "geometry_unchanged": True, "measured_albedo": False, "measured_brdf": False,
        "camera_to_mesh_registration": False, "nonrigid_surface_tracking": False,
        "endoscope_rendered": False, "clinical_visual_validation": False,
        "privacy": "Source videos, extracted frames, median patches and textures remain local and git-ignored.",
        "limitations": [
            "ROIs are manually selected and not instance-segmented or anatomically verified",
            "Translation-only alignment cannot follow nonrigid tissue motion or changing view angle",
            "A temporal median can suppress transient highlights but does not recover intrinsic albedo",
            "Quilting still synthesizes repeated local appearance and is not image-to-3D registration",
            "Isaac endoscope render and deforming-mesh review remain required before visual acceptance",
        ],
    }
    if sources:
        from pxr import Usd
        from reference_appearance import author_texture
        stage = Usd.Stage.CreateNew(str(out_dir / "appearance.usda"))
        for source in map(Path, sources):
            source = source.resolve()
            stage.GetRootLayer().subLayerPaths.append(
                os.path.relpath(source, out_dir).replace("\\", "/"))
        targets = {"appendix": "/World/Atlas/appendix/render",
                   "mesoappendix": "/World/Atlas/mesoappendix/render"}
        records = []
        for name, prim in targets.items():
            if not stage.GetPrimAtPath(prim):
                raise ValueError(f"Target mesh is absent from source scene: {prim}")
            records.append(author_texture(stage, prim,
                f"{name}_appendix300_temporal_rgb.png", repeat=1., roughness=.34,
                material_path=f"/World/Appendix300Look/{name}"))
        stage.GetRootLayer().Save()
        report["mapped_meshes"] = records
        report["source_scene_sha256"] = [hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                         for p in sources]
    if parent_layer:
        report["static_usd_layer"] = rebind_reviewed_uv_layer(parent_layer, out_dir)
    (out_dir / "appearance.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(".cache/bin/ffmpeg.exe"))
    parser.add_argument("--center-seconds", type=float, default=12.)
    parser.add_argument("--frame-count", type=int, default=9)
    parser.add_argument("--spacing-seconds", type=float, default=.06)
    parser.add_argument("--source", type=Path, nargs="+")
    parser.add_argument("--parent-layer", type=Path,
                        help="Rebind a reviewed static USD UV layer without changing UVs")
    args = parser.parse_args()
    if args.source and args.parent_layer:
        parser.error("--source and --parent-layer are mutually exclusive")
    result = build(args.reference, args.out, args.ffmpeg, args.center_seconds,
                   args.frame_count, args.spacing_seconds, args.source, args.parent_layer)
    print(json.dumps(result, indent=2))
