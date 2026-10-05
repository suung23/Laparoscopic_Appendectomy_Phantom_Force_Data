"""Summarize archived PhysX jaw-effort and tissue-health event timing.

Joint effort and locally tared mount residuals are simulator measurements,
not calibrated external tissue force. Source trajectories stay in ignored out/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIALS = (
    "contact_baseline_v1",
    "contact_probe_v1",
    "contact_surface_v1",
    "contact_surface_v2",
    "contact_surface_v3",
    "contact_free_control_v1",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def summarize_trace(report, trace, jaw_trip_nm=0.0003):
    """Summarize event ordering without fitting a force scale or time shift."""
    if not isinstance(trace, list) or not trace:
        raise ValueError("A non-empty per-physics-tick trace is required")
    frequency = _number(report.get("physics_hz"), "physics_hz")
    threshold = _number(jaw_trip_nm, "jaw_trip_nm")
    if frequency <= 0 or threshold <= 0:
        raise ValueError("Physics frequency and jaw trip threshold must be positive")
    if int(report.get("physics_ticks_checked", -1)) != len(trace):
        raise ValueError("Physics trace length does not match report tick count")

    arms = {}
    first_inversion = None
    minimum_ratio = math.inf
    maximum_inverted = 0
    for hand in ("left", "right"):
        efforts = []
        mounts = []
        limited = []
        for tick, sample in enumerate(trace):
            try:
                values = sample["contact"][hand]["jaw_effort_nm"]
                residual = sample["contact"][hand]["residual_force_n"]
                is_limited = bool(sample["contact"][hand]["jaw_limited"])
                if len(values) != 2:
                    raise ValueError("jaw_effort_nm must have two entries")
                efforts.append(max(abs(_number(v, f"{hand}.jaw_effort_nm")) for v in values))
                mounts.append(_number(residual, f"{hand}.residual_force_n"))
                if is_limited:
                    limited.append(tick)
                ratio = _number(sample["tissue_health"]["minimum_signed_volume_ratio"],
                                "minimum_signed_volume_ratio")
                inverted_value = _number(sample["tissue_health"]["inverted_tetrahedra"],
                                         "inverted_tetrahedra")
                if inverted_value < 0 or not inverted_value.is_integer():
                    raise ValueError(f"Invalid inverted tetrahedron count at physics tick {tick}")
                inverted = int(inverted_value)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Malformed contact/tissue sample at physics tick {tick}") from exc
            minimum_ratio = min(minimum_ratio, ratio)
            maximum_inverted = max(maximum_inverted, inverted)
            if inverted > 0 and first_inversion is None:
                first_inversion = tick

        trip = next((i for i, value in enumerate(efforts) if value >= threshold), None)
        limit_tick = limited[0] if limited else None
        arms[hand] = {
            "peak_absolute_jaw_effort_nm": max(efforts),
            "jaw_effort_trip_threshold_nm": threshold,
            "first_threshold_crossing_tick": trip,
            "first_threshold_crossing_trace_time_s": None if trip is None else trip / frequency,
            "first_jaw_limit_tick": limit_tick,
            "first_jaw_limit_trace_time_s": None if limit_tick is None else limit_tick / frequency,
            "peak_locally_tared_mount_residual_n": max(mounts),
            "mount_residual_is_calibrated_tissue_force": False,
        }

    events = [arms[h]["first_threshold_crossing_tick"] for h in arms
              if arms[h]["first_threshold_crossing_tick"] is not None]
    first_jaw_trip = min(events) if events else None
    trip_precedes_inversion = None if first_jaw_trip is None or first_inversion is None else first_jaw_trip <= first_inversion
    return {
        "report_status": report.get("status"),
        "reported_failure": report.get("failure", ""),
        "physics_frequency_hz": frequency,
        "trace_time_convention": "zero-origin at the first saved physics tick; excludes reset/settling; one-tick timing resolution",
        "trace_samples": len(trace),
        "minimum_signed_volume_ratio": minimum_ratio,
        "maximum_inverted_tetrahedra": maximum_inverted,
        "first_inversion_tick": first_inversion,
        "first_inversion_trace_time_s": None if first_inversion is None else first_inversion / frequency,
        "first_any_jaw_threshold_crossing_tick": first_jaw_trip,
        "jaw_effort_trip_precedes_or_matches_first_inversion": trip_precedes_inversion,
        "arms": arms,
        "measurement_scope": [
            "Jaw effort is actuator-joint effort, not external tissue force.",
            "Mount residual is locally tared joint reaction, not calibrated tissue wrench.",
            "Tissue health is sampled on saved physics ticks; it is numerical mesh integrity, not biological injury.",
        ],
    }


def analyze(root=ROOT, manifest_path=None, trials=DEFAULT_TRIALS, jaw_trip_nm=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path or root / "validation/ct-contact-feedback.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    threshold = float(jaw_trip_nm if jaw_trip_nm is not None else manifest["jaw_trip_nm"])
    results = {}
    for name in trials:
        expected = manifest["runs"].get(name)
        if expected is None:
            raise ValueError(f"No provenance hash in source manifest for trial {name}")
        directory = root / "out/ct_appendectomy" / name
        report_path, trace_path = directory / "report.json", directory / "physics_trace.json"
        if sha256(report_path) != expected["report_sha256"]:
            raise ValueError(f"Report hash mismatch for {name}")
        if sha256(trace_path) != expected["trace_sha256"]:
            raise ValueError(f"Physics trace hash mismatch for {name}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        results[name] = {
            "source_report_sha256": expected["report_sha256"],
            "source_trace_sha256": expected["trace_sha256"],
            "summary": summarize_trace(report, trace, threshold),
        }
    return {
        "status": "hash-verified force/deformation trace analysis; no biological calibration",
        "source_manifest": "validation/ct-contact-feedback.json",
        "jaw_effort_trip_threshold_nm": threshold,
        "trial_count": len(results),
        "trials": results,
        "limitations": [
            "All traces are prior local Isaac runs; this analysis does not rerun or validate the current executable.",
            "Locally tared mount reactions and actuator-joint efforts are not calibrated tissue force.",
            "A jaw-effort threshold preceding inversion does not prove absence of sub-inversion tissue damage.",
            "One contact site was tested on one fixed anatomy; right-side contact and bimanual grasp remain unvalidated.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--trials", nargs="+", default=DEFAULT_TRIALS)
    parser.add_argument("--jaw-trip-nm", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.manifest, args.trials, args.jaw_trip_nm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
