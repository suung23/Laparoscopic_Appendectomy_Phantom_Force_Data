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
    return (rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114) / 255.


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


def align_dense(reference, moving):
    """Nonrigid TV-L1 warp plus forward/backward consistency mask."""
    try:
        from scipy.ndimage import map_coordinates
        from skimage.registration import optical_flow_tvl1
    except ImportError as error:
        raise RuntimeError("Dense alignment requires the installed scipy and scikit-image packages") from error
    ref_gray, mov_gray = _gray(reference), _gray(moving)
    if ref_gray.shape != mov_gray.shape:
        raise ValueError("Dense-flow inputs must have matching dimensions")
    v, u = optical_flow_tvl1(ref_gray, mov_gray, attachment=15, tightness=.3,
                             num_warp=5, num_iter=10, tol=1e-4, prefilter=True)
    back_v, back_u = optical_flow_tvl1(mov_gray, ref_gray, attachment=15, tightness=.3,
                                       num_warp=5, num_iter=10, tol=1e-4, prefilter=True)
    h, w = ref_gray.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    sample_y, sample_x = y + v, x + u
    coords = np.stack((sample_y, sample_x))
    warped = np.stack([map_coordinates(np.asarray(moving)[..., channel], coords,
                                       order=1, mode="nearest")
                       for channel in range(3)], axis=-1)
    rev_v = map_coordinates(back_v, coords, order=1, mode="nearest")
    rev_u = map_coordinates(back_u, coords, order=1, mode="nearest")
    roundtrip = np.hypot(v + rev_v, u + rev_u)
    valid = ((sample_y >= 1) & (sample_y < h - 1) &
             (sample_x >= 1) & (sample_x < w - 1) & (roundtrip <= 1.5))
    flow_magnitude = np.hypot(v, u)
    return Image.fromarray(np.uint8(np.clip(warped, 0, 255))), valid, {
        "method": "skimage_TVL1_dense_flow",
        "flow_magnitude_p95_px": round(float(np.percentile(flow_magnitude, 95)), 3),
        "flow_magnitude_max_px": round(float(flow_magnitude.max()), 3),
        "forward_backward_consistent_fraction": round(float(valid.mean()), 5),
        "roundtrip_tolerance_px": 1.5,
    }


def fuse_temporal_detail(aligned, valid_masks, reference_index, *, sigma_px=2.0,
                         consistency_threshold_rgb=8.0, detail_gain=1.0):
    """Restore reference detail only where neighboring frames support it.

    Dense-flow warping can soften detail even when its large-scale alignment is
    useful. This keeps the median low-frequency appearance but selectively
    restores high-frequency detail from the unwarped reference where neighboring
    frames support similar detail. It remains an appearance candidate, not an
    albedo or 3-D surface estimate.
    """
    from scipy.ndimage import gaussian_filter

    stack = np.asarray(aligned, dtype=np.float32)
    valid = np.asarray(valid_masks, dtype=bool)
    if (stack.ndim != 4 or stack.shape[-1] != 3 or valid.shape != stack.shape[:3] or
            stack.shape[0] < 3 or not 0 <= reference_index < stack.shape[0] or
            not np.isfinite(stack).all() or not np.isfinite(sigma_px) or sigma_px <= 0 or
            not np.isfinite(consistency_threshold_rgb) or consistency_threshold_rgb <= 0 or
            not np.isfinite(detail_gain) or not 0 <= detail_gain <= 2):
        raise ValueError("Invalid temporal detail-fusion inputs")
    low = gaussian_filter(stack, sigma=(0, sigma_px, sigma_px, 0), mode="reflect")
    detail = stack - low
    count = valid.sum(axis=0)
    masked_low = np.where(valid[..., None], low, np.nan)
    masked_detail = np.where(valid[..., None], detail, np.nan)
    no_support = count == 0
    masked_low[reference_index, no_support] = low[reference_index, no_support]
    masked_detail[reference_index, no_support] = detail[reference_index, no_support]
    with np.errstate(all="ignore"):
        fused_low = np.nanmedian(masked_low, axis=0)
        median_detail = np.nanmedian(masked_detail, axis=0)
        detail_error = np.abs(masked_detail - median_detail[None, ...])
        detail_error = np.nan_to_num(detail_error, nan=0.0, posinf=0.0)
        per_frame_error = detail_error.mean(axis=-1)
        per_frame_error[~valid] = np.nan
        disagreement = np.nanmedian(per_frame_error, axis=0)
    supported = count >= max(3, (stack.shape[0] + 1) // 2)
    ref_low = low[reference_index]
    ref_detail = detail[reference_index]
    fused_low[~supported] = ref_low[~supported]
    median_detail[~supported] = ref_detail[~supported]
    disagreement = np.nan_to_num(disagreement, nan=consistency_threshold_rgb * 2,
                                 posinf=consistency_threshold_rgb * 2)
    confidence = np.clip(1.0 - disagreement / consistency_threshold_rgb, 0.0, 1.0)
    confidence = gaussian_filter(confidence, sigma=.75, mode="nearest")
    confidence[~supported] = 0.0
    fused_detail = ((1.0 - confidence[..., None]) * median_detail +
                    confidence[..., None] * ref_detail)
    fused = np.uint8(np.clip(fused_low + detail_gain * fused_detail, 0, 255))
    supported_channels = np.broadcast_to(supported[..., None], fused_detail.shape)

    def rms(values):
        sample = values[supported_channels]
        return float(np.sqrt(np.mean(np.square(sample)))) if sample.size else None

    return fused, {
        "method": "temporal_median_low_frequency_with_consensus_gated_reference_detail",
        "sigma_px": float(sigma_px),
        "consistency_threshold_rgb": float(consistency_threshold_rgb),
        "detail_gain": float(detail_gain),
        "supported_fraction": float(supported.mean()),
        "low_confidence_fraction": float((supported & (confidence < .2)).sum() /
                                          max(int(supported.sum()), 1)),
        "reference_detail_rms_rgb": rms(ref_detail),
        "temporal_median_detail_rms_rgb": rms(median_detail),
        "fused_detail_rms_rgb": rms(fused_detail),
        "interpretation": "Frequency-preserving appearance candidate; no albedo, BRDF, or 3-D registration inferred",
    }


def deduplicate_odd_window(frame_records, center_seconds):
    """Remove exact duplicate decodes and retain an odd unique frame window."""
    unique, duplicate_frames, hashes_seen = [], [], {}
    for record in frame_records:
        digest = record["sha256"]
        if digest in hashes_seen:
            duplicate_frames.append({
                "requested_time_seconds": record["requested_time_seconds"],
                "decoded_time_seconds": record["time_seconds"],
                "duplicate_of_requested_time_seconds": hashes_seen[digest],
                "sha256": digest,
            })
        else:
            hashes_seen[digest] = record["requested_time_seconds"]
            unique.append(record)
    discarded = []
    if len(unique) % 2 == 0:
        farthest = max(range(len(unique)),
                       key=lambda index: abs(unique[index]["time_seconds"] - center_seconds))
        removed = unique.pop(farthest)
        discarded.append({"requested_time_seconds": removed["requested_time_seconds"],
                          "decoded_time_seconds": removed["time_seconds"],
                          "reason": "trimmed farthest unique frame to retain an odd temporal window",
                          "sha256": removed["sha256"]})
    if len(unique) < 3:
        raise ValueError("Fewer than three unique source frames remain after duplicate removal")
    unique.sort(key=lambda record: record["time_seconds"])
    return unique, {"duplicate_decodes": duplicate_frames,
                    "trimmed_for_odd_window": discarded}


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


def copy_reviewed_temporal_uv_layer(parent_layer, output_dir):
    """Copy a hash-verified temporal UV layer beside replacement RGB textures.

    This preserves a prior layer byte-for-byte; it does not edit mesh topology,
    UVs, material parameters, or physics. Relative source layers must resolve
    and still match the recorded hashes from the output directory.
    """
    parent_layer = Path(parent_layer).resolve()
    output_dir = Path(output_dir).resolve()
    parent_report_path = parent_layer.with_name("appearance.json")
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    layer_report = parent_report.get("static_usd_layer", {})
    digest = hashlib.sha256(parent_layer.read_bytes()).hexdigest()
    if (parent_report.get("dataset") != DATASET or parent_report.get("license") != "CC BY 4.0" or
            parent_report.get("geometry_unchanged") is not True or
            layer_report.get("sha256") != digest or
            layer_report.get("uvs_reused_unchanged") is not True or
            layer_report.get("geometry_unchanged_by_layer_operation") is not True or
            layer_report.get("physics_unchanged_by_layer_operation") is not True):
        raise ValueError("Parent temporal USD layer lacks matching provenance and preservation checks")
    text = parent_layer.read_text(encoding="utf-8")
    for name in ("appendix", "mesoappendix"):
        if re.search(rf'over "{name}"\s*\{{\s*over "render"', text) is None:
            raise ValueError(f"Parent temporal USD layer does not bind expected mesh: {name}")
    for name in ("appendix", "mesoappendix"):
        asset = f"@{name}_appendix300_temporal_rgb.png@"
        if text.count(asset) != 1 or not (output_dir / asset.strip("@")).is_file():
            raise ValueError(f"Expected one replacement temporal texture asset: {name}")
    records = layer_report.get("source_scene_layers", [])
    sources = layer_report.get("referenced_sublayers", [])
    if len(records) != len(sources):
        raise ValueError("Parent report does not hash every temporal USD sublayer")
    sublayer_block = re.search(r"subLayers\s*=\s*\[(.*?)\]", text, flags=re.S)
    actual_sources = (re.findall(r"@([^@]+)@", sublayer_block.group(1))
                      if sublayer_block is not None else [])
    if actual_sources != sources:
        raise ValueError("Temporal USD sublayer list differs from the provenance report")
    resolved_sources = []
    for relative, record in zip(sources, records):
        if record.get("path") != relative:
            raise ValueError("Temporal USD sublayer path differs from the provenance report")
        source = (output_dir / relative).resolve()
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError(f"Temporal USD sublayer changed or is missing: {relative}")
        resolved_sources.append({"path": relative, "sha256": record["sha256"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "appearance.usda"
    target.write_text(text, encoding="utf-8")
    return {
        "path": target.name,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "parent_layer_sha256": digest,
        "referenced_sublayers": sources,
        "source_scene_layers": resolved_sources,
        "changed_properties": [],
        "uvs_reused_unchanged": True,
        "geometry_unchanged_by_layer_operation": True,
        "physics_unchanged_by_layer_operation": True,
        "usd_runtime_load_verified": False,
    }


def extract_frames(video, ffmpeg, output, times):
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    expected_size = None
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
            if expected_size is None:
                expected_size = image.size
            elif image.size != expected_size:
                raise ValueError(f"Frame dimensions changed at {timestamp}s: {image.size}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = {"time_seconds": actual_timestamp, "requested_time_seconds": timestamp,
                  "path": path, "sha256": digest}
        frames.append(record)
    frames, selection = deduplicate_odd_window(frames, float(np.mean(times)))
    return frames, expected_size, selection


def build(reference_dir, out_dir, ffmpeg, center_seconds=12., frame_count=9,
          spacing_seconds=.06, sources=None, parent_layer=None,
          source_video_name="Center1_014.mp4", roi_overrides=None,
          alignment_method="phase_translation", texture_strategy="median"):
    reference_dir, out_dir = Path(reference_dir).resolve(), Path(out_dir).resolve()
    ffmpeg = Path(ffmpeg).resolve()
    provenance_path = reference_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("doi") != "10.25532/OPARA-1173" or provenance.get("license") != "CC BY 4.0":
        raise ValueError("Local source manifest is not the licensed Appendix300 record")
    item = next((f for f in provenance["files"] if f["local_name"] == source_video_name), None)
    if item is None:
        raise ValueError(f"Source manifest does not include {source_video_name}")
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
    if roi_overrides is None:
        if source_video_name != "Center1_014.mp4":
            raise ValueError("A visually reviewed --roi is required for non-default source videos")
        roi_boxes = dict(ROIS)
    else:
        roi_boxes = dict(roi_overrides)
    if set(roi_boxes) != {"appendix", "mesoappendix"}:
        raise ValueError("ROI names must be exactly appendix and mesoappendix")
    if alignment_method not in {"phase_translation", "dense_flow_tvl1"}:
        raise ValueError("Unsupported alignment method")
    if texture_strategy not in {"median", "detail_fused"}:
        raise ValueError("Unsupported texture strategy")
    if texture_strategy == "detail_fused" and alignment_method != "dense_flow_tvl1":
        raise ValueError("detail_fused requires dense_flow_tvl1 alignment")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_records, frame_size, frame_selection = extract_frames(
        video, ffmpeg, out_dir / "frames", times)
    full_frames = []
    for record in frame_records:
        with Image.open(record["path"]) as im:
            full_frames.append((record["time_seconds"], im.convert("RGB")))

    rois, alignment_report = {}, {}
    for index, (name, box) in enumerate(roi_boxes.items()):
        if len(box) != 4 or box[0] < 0 or box[1] < 0 or box[2] > frame_size[0] or box[3] > frame_size[1] or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"ROI {name} is outside {frame_size}: {box}")
        crops = [(timestamp, image.crop(box)) for timestamp, image in full_frames]
        reference_index = len(crops) // 2
        reference = crops[reference_index][1]
        aligned, valid_masks, transforms, shifts = [], [], [], []
        for frame_index, (timestamp, crop) in enumerate(crops):
            if alignment_method == "dense_flow_tvl1" and frame_index != reference_index:
                aligned_crop, valid, record = align_dense(reference, crop)
                aligned.append(np.asarray(aligned_crop, dtype=np.uint8))
                valid_masks.append(valid)
                transforms.append({"time_seconds": timestamp, **record})
            else:
                if alignment_method == "dense_flow_tvl1":
                    valid_masks.append(np.ones((reference.height, reference.width), dtype=bool))
                    transforms.append({"time_seconds": timestamp, "method": "reference_frame",
                                       "forward_backward_consistent_fraction": 1.0})
                else:
                    dx, dy, ratio = phase_translation(reference, crop)
                    shifts.append((dx, dy))
                    transforms.append({"time_seconds": timestamp, "sample_shift_xy_px": [dx, dy],
                                       "peak_to_sidelobe_ratio": round(ratio, 5)})
                aligned.append(np.asarray(crop if alignment_method == "dense_flow_tvl1"
                                          else translate(crop, dx, dy), dtype=np.uint8))
        if alignment_method == "dense_flow_tvl1":
            stack = np.stack(aligned, axis=0).astype(np.float32)
            valid = np.stack(valid_masks, axis=0)
            valid_count = valid.sum(axis=0)
            masked = np.where(valid[..., None], stack, np.nan)
            with np.errstate(all="ignore"):
                temporal = np.nanmedian(masked, axis=0)
            underconstrained = valid_count < 2
            temporal[underconstrained] = stack[len(stack) // 2][underconstrained]
            temporal = np.uint8(np.clip(np.nan_to_num(temporal), 0, 255))
            pad_x = pad_y = 0
            transform_info = {"method": alignment_method,
                              "pixels_with_two_or_more_valid_frames_fraction": round(
                                  float((valid_count >= 2).mean()), 5)}
        else:
            temporal = np.median(np.stack(aligned, axis=0), axis=0).astype(np.uint8)
            # Exclude replicated border pixels so only observed texels enter the median.
            pad_x = max(abs(dx) for dx, _ in shifts)
            pad_y = max(abs(dy) for _, dy in shifts)
            if pad_x * 2 >= temporal.shape[1] or pad_y * 2 >= temporal.shape[0]:
                raise ValueError("Alignment leaves no common observed area in the ROI")
            temporal = temporal[pad_y:temporal.shape[0] - pad_y if pad_y else None,
                                pad_x:temporal.shape[1] - pad_x if pad_x else None]
            transform_info = {"method": alignment_method}
        temporal_path = out_dir / f"{name}_temporal_median.png"
        Image.fromarray(temporal).save(temporal_path)
        texture_path = out_dir / f"{name}_appendix300_temporal_rgb.png"
        # Keep an un-repeated crop as the primary candidate. Its UV scale is
        # still approximate, but this avoids fabricating a tiled surface field.
        source_patch = Image.fromarray(temporal)
        detail_info = None
        if texture_strategy == "detail_fused":
            fused, detail_info = fuse_temporal_detail(aligned, valid_masks, reference_index)
            source_patch = Image.fromarray(fused)
            source_patch.save(out_dir / f"{name}_temporal_detail_fused.png")
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
        if detail_info is not None:
            rois[name]["texture_method"] = "consensus_gated_reference_detail_over_temporal_median"
            rois[name]["detail_fusion"] = detail_info
        alignment_report[name] = {"summary": transform_info, "per_frame": transforms}

    if parent_layer:
        status = "static_usd_candidate_runtime_unverified"
    elif sources:
        status = "multi_frame_usd_candidate_unrendered"
    else:
        status = "multi_frame_texture_candidate_usd_pending"
    report = {
        "status": status,
        "dataset": DATASET, "attribution": ATTRIBUTION,
        "license": provenance["license"], "source_doi": provenance["doi"],
        "source_video": item["name"], "source_video_sha256": video_hash,
        "source_frame_dimensions_px": list(frame_size),
        "requested_frame_times_seconds": times,
        "source_frames": [{"decoded_time_seconds": record["time_seconds"],
                            "requested_time_seconds": record["requested_time_seconds"],
                            "sha256": record["sha256"]} for record in frame_records],
        "frame_selection": frame_selection,
        "alignment_method": alignment_method,
        "texture_strategy": texture_strategy,
        "alignment_by_roi": alignment_report, "rois_and_textures": rois,
        "texture_mapping": "Each temporal composite crop fills one UV map; no 3-D camera projection or patient correspondence",
        "geometry_unchanged": True, "measured_albedo": False, "measured_brdf": False,
        "camera_to_mesh_registration": False,
        "dense_2d_optical_flow_used": alignment_method == "dense_flow_tvl1",
        "physical_nonrigid_surface_tracking": False,
        "endoscope_rendered": False, "clinical_visual_validation": False,
        "privacy": "Source videos, extracted frames, median patches and textures remain local and git-ignored.",
        "limitations": [
            "ROIs are manually selected and not instance-segmented or anatomically verified",
            ("Dense flow tracks 2-D image motion only; it does not estimate 3-D deformation or surface attachment"
             if alignment_method == "dense_flow_tvl1" else
             "Translation-only alignment cannot follow nonrigid tissue motion or changing view angle"),
            "A temporal median can reduce transient highlights but does not recover intrinsic albedo or BRDF",
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
        report["static_usd_layer"] = (
            copy_reviewed_temporal_uv_layer(parent_layer, out_dir)
            if texture_strategy == "detail_fused" else
            rebind_reviewed_uv_layer(parent_layer, out_dir))
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
    parser.add_argument("--source-video", default="Center1_014.mp4",
                        help="Local filename exactly as recorded in provenance.json")
    parser.add_argument("--roi", action="append", metavar="NAME=X1,Y1,X2,Y2",
                        help="Reviewed tissue crop; repeat once for appendix and mesoappendix")
    parser.add_argument("--alignment-method", choices=("phase_translation", "dense_flow_tvl1"),
                        default="phase_translation")
    parser.add_argument("--texture-strategy", choices=("median", "detail_fused"), default="median",
                        help="detail_fused requires dense flow; output remains an unreviewed candidate")
    parser.add_argument("--source", type=Path, nargs="+")
    parser.add_argument("--parent-layer", type=Path,
                        help="Rebind a reviewed static USD UV layer without changing UVs")
    args = parser.parse_args()
    if args.source and args.parent_layer:
        parser.error("--source and --parent-layer are mutually exclusive")
    roi_overrides = None
    if args.roi:
        roi_overrides = {}
        try:
            for item in args.roi:
                name, coordinate_text = item.split("=", 1)
                coords = tuple(int(value) for value in coordinate_text.split(","))
                if name in roi_overrides or len(coords) != 4:
                    raise ValueError
                roi_overrides[name] = coords
        except ValueError:
            parser.error("Each --roi must be NAME=X1,Y1,X2,Y2 with unique names")
    result = build(args.reference, args.out, args.ffmpeg, args.center_seconds,
                   args.frame_count, args.spacing_seconds, args.source, args.parent_layer,
                   args.source_video, roi_overrides, args.alignment_method,
                   args.texture_strategy)
    print(json.dumps(result, indent=2))
