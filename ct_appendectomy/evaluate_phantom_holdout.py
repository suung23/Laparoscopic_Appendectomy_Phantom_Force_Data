"""Score a frozen simulator against group-held-out phantom trajectories.

This reports agreement metrics only. It does not calibrate parameters, declare
acceptance thresholds, or establish biological validity.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED = {
    "split",
    "specimen_or_build_id",
    "trajectory_id",
    "time_s",
    "phase",
    "force_measured_x_N",
    "force_measured_y_N",
    "force_measured_z_N",
    "force_predicted_x_N",
    "force_predicted_y_N",
    "force_predicted_z_N",
    "displacement_measured_mm",
    "displacement_predicted_mm",
    "contact_measured",
    "contact_predicted",
}
SPLITS = {"development", "heldout"}


def _number(row: dict[str, str], key: str, line: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"line {line}: {key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"line {line}: {key} must be finite")
    return value


def _binary(row: dict[str, str], key: str, line: int) -> int:
    value = _number(row, key, line)
    if value not in (0.0, 1.0):
        raise ValueError(f"line {line}: {key} must be 0 or 1")
    return int(value)


def _vector(row: dict[str, str], prefix: str, line: int) -> tuple[float, float, float]:
    return tuple(_number(row, f"{prefix}_{axis}_N", line) for axis in "xyz")


def _cluster_bootstrap_ci(
    groups: dict[str, list[dict[str, object]]],
    squared_error,
    replicates: int = 2000,
) -> dict[str, object] | None:
    """Percentile CI by resampling whole heldout specimens/builds, not frames."""
    names = sorted(name for name, values in groups.items() if values)
    if len(names) < 2:
        return None
    rng = random.Random(1729)
    estimates = []
    for _ in range(replicates):
        draw = [rng.choice(names) for _ in names]
        errors = [squared_error(item) for name in draw for item in groups[name]]
        estimates.append(math.sqrt(sum(errors) / len(errors)))
    estimates.sort()
    return {
        "lower_95": estimates[math.floor(.025 * (replicates - 1))],
        "upper_95": estimates[math.ceil(.975 * (replicates - 1))],
        "heldout_groups": len(names),
        "replicates": replicates,
        "method": "specimen/build cluster percentile bootstrap",
    }


def score(rows: Iterable[dict[str, str]]) -> dict[str, object]:
    records = list(rows)
    if not records:
        raise ValueError("input contains no trajectory rows")
    groups: dict[str, str] = {}
    trajectories: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for line, row in enumerate(records, start=2):
        missing = REQUIRED.difference(row)
        if missing:
            raise ValueError(f"line {line}: missing columns {', '.join(sorted(missing))}")
        split = row["split"].strip()
        group = row["specimen_or_build_id"].strip()
        trajectory = row["trajectory_id"].strip()
        if split not in SPLITS or not group or not trajectory:
            raise ValueError(f"line {line}: invalid split or empty group/trajectory ID")
        previous = groups.setdefault(group, split)
        if previous != split:
            raise ValueError("a specimen/build occurs in both development and heldout splits")
        item: dict[str, object] = {
            "time": _number(row, "time_s", line),
            "phase": row["phase"].strip().lower(),
            "fm": _vector(row, "force_measured", line),
            "fp": _vector(row, "force_predicted", line),
            "dm": _number(row, "displacement_measured_mm", line),
            "dp": _number(row, "displacement_predicted_mm", line),
            "cm": _binary(row, "contact_measured", line),
            "cp": _binary(row, "contact_predicted", line),
        }
        trajectories[(group, trajectory)].append(item)

    if not {"development", "heldout"}.issubset(set(groups.values())):
        raise ValueError("input must include development and heldout specimen/build groups")
    heldout = {key: sorted(value, key=lambda x: x["time"]) for key, value in trajectories.items() if groups[key[0]] == "heldout"}
    for values in heldout.values():
        times = [float(x["time"]) for x in values]
        if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("each heldout trajectory needs at least two strictly increasing timestamps")

    samples = [item for values in heldout.values() for item in values]
    group_samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (group, _), values in heldout.items():
        group_samples[group].extend(values)

    def rmse(pairs: list[tuple[float, float]]) -> float | None:
        return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs)) if pairs else None

    def force_metrics(items: list[dict[str, object]]) -> dict[str, object]:
        if not items:
            return {"vector_rmse_N": None, "p95_vector_error_N": None, "max_vector_error_N": None, "bias_xyz_N": None}
        errors = [tuple(float(a) - float(b) for a, b in zip(x["fm"], x["fp"])) for x in items]
        norms = [math.sqrt(sum(component * component for component in error)) for error in errors]
        return {
            "vector_rmse_N": math.sqrt(sum(value * value for value in norms) / len(norms)),
            "p95_vector_error_N": sorted(norms)[math.ceil(.95 * len(norms)) - 1],
            "max_vector_error_N": max(norms),
            "bias_xyz_N": [sum(error[i] for error in errors) / len(errors) for i in range(3)],
        }

    def force_error_squared(item: dict[str, object]) -> float:
        return sum((float(a) - float(b)) ** 2 for a, b in zip(item["fm"], item["fp"]))

    displacement_groups = {
        group: items for group, items in group_samples.items()
    }
    relaxation_groups = {
        group: [item for item in items if item["phase"] == "relaxation"]
        for group, items in group_samples.items()
    }

    by_phase: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in samples:
        by_phase[str(item["phase"])].append(item)
    transition_errors: list[float] = []
    transition_count_error = 0
    transition_direction_mismatches = 0
    for values in heldout.values():
        measured = [i for i in range(1, len(values)) if values[i]["cm"] != values[i - 1]["cm"]]
        predicted = [i for i in range(1, len(values)) if values[i]["cp"] != values[i - 1]["cp"]]
        transition_count_error += abs(len(measured) - len(predicted))
        for a, b in zip(measured, predicted):
            transition_errors.append(abs(float(values[a]["time"]) - float(values[b]["time"])))
            if (values[a - 1]["cm"], values[a]["cm"]) != (values[b - 1]["cp"], values[b]["cp"]):
                transition_direction_mismatches += 1

    return {
        "status": "agreement metrics only; no pass/fail threshold",
        "time_alignment": "input prediction values must already be sampled at measured timestamps; no fitted time shift or force scale",
        "groups": {"development": sum(v == "development" for v in groups.values()), "heldout": sum(v == "heldout" for v in groups.values())},
        "heldout_trajectory_count": len(heldout),
        "heldout_sample_count": len(samples),
        "heldout_metrics": {
            **force_metrics(samples),
            "displacement_rmse_mm": rmse([(float(x["dm"]), float(x["dp"])) for x in samples]),
            "relaxation_force": force_metrics(by_phase.get("relaxation", [])),
            "contact_transition_count_error": transition_count_error,
            "contact_transition_direction_mismatches": transition_direction_mismatches,
            "contact_transition_time_mae_s": sum(transition_errors) / len(transition_errors) if transition_errors else None,
            "contact_transition_pairs_scored": len(transition_errors),
        },
        "heldout_rmse_95ci": {
            "force_vector_N": _cluster_bootstrap_ci(group_samples, force_error_squared),
            "displacement_mm": _cluster_bootstrap_ci(
                displacement_groups,
                lambda x: (float(x["dm"]) - float(x["dp"])) ** 2,
            ),
            "relaxation_force_vector_N": _cluster_bootstrap_ci(relaxation_groups, force_error_squared),
        },
        "uncertainty_note": "cluster bootstrap resamples whole heldout specimens/builds; CI is null with fewer than two eligible groups",
        "limitations": [
            "parameters must be frozen before scoring heldout groups",
            "thresholds must be preregistered from instrument repeatability and specimen variability",
            "this score does not demonstrate policy transfer or biological tissue fidelity",
        ],
    }


def self_test() -> None:
    sample = """split,specimen_or_build_id,trajectory_id,time_s,phase,force_measured_x_N,force_measured_y_N,force_measured_z_N,force_predicted_x_N,force_predicted_y_N,force_predicted_z_N,displacement_measured_mm,displacement_predicted_mm,contact_measured,contact_predicted
development,dev-1,t-1,0,loading,0,0,0,0,0,0,0,0,0,0
development,dev-1,t-1,1,loading,1,0,0,1,0,0,1,1,1,1
heldout,test-1,t-2,0,loading,0,0,0,0,0,0,0,0,0,0
heldout,test-1,t-2,1,loading,2,0,0,3,0,0,1,1.5,1,1
heldout,test-1,t-2,2,relaxation,1,0,0,2,0,0,2,2,1,1
"""
    result = score(csv.DictReader(io.StringIO(sample)))
    metrics = result["heldout_metrics"]
    assert result["groups"] == {"development": 1, "heldout": 1}
    assert math.isclose(float(metrics["vector_rmse_N"]), math.sqrt(2 / 3))
    assert math.isclose(float(metrics["displacement_rmse_mm"]), math.sqrt(0.25 / 3))
    assert metrics["contact_transition_count_error"] == 0
    assert metrics["contact_transition_direction_mismatches"] == 0
    assert result["heldout_rmse_95ci"]["force_vector_N"] is None
    leaking = list(csv.DictReader(io.StringIO(sample)))
    leaking.append(dict(leaking[0], split="heldout", trajectory_id="leak", time_s="2"))
    try:
        score(leaking)
    except ValueError as exc:
        assert "both" in str(exc)
    else:
        raise AssertionError("specimen/build leakage must be rejected")
    multi_group = list(csv.DictReader(io.StringIO(sample)))
    for row in list(multi_group):
        if row["split"] == "heldout":
            duplicate = dict(row, specimen_or_build_id="test-2", trajectory_id="t-3")
            for axis in "xyz":
                duplicate[f"force_predicted_{axis}_N"] = duplicate[f"force_measured_{axis}_N"]
            duplicate["displacement_predicted_mm"] = duplicate["displacement_measured_mm"]
            multi_group.append(duplicate)
    multi_result = score(multi_group)
    interval = multi_result["heldout_rmse_95ci"]["force_vector_N"]
    assert interval is not None and interval["lower_95"] <= interval["upper_95"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true", help="run a synthetic unit check; this is not phantom evidence")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("synthetic self-test passed; no physical phantom data evaluated")
        return
    if args.csv_path is None:
        parser.error("csv_path is required unless --self-test is used")
    with args.csv_path.open(newline="", encoding="utf-8-sig") as stream:
        print(json.dumps(score(csv.DictReader(stream)), indent=2))


if __name__ == "__main__":
    main()
