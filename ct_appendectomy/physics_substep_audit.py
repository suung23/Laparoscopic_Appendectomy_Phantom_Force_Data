"""Fail-closed summaries for tissue geometry sampled at every underlying solver step."""
from __future__ import annotations

import math


def summarize_interval(samples, physics_dt_s):
    """Summarize compact ``tetra_health`` dictionaries for one environment action."""
    if not samples:
        raise ValueError("A physics interval must contain at least one solver-step sample")
    try:
        dt = float(physics_dt_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("physics_dt_s must be finite and positive") from exc
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("physics_dt_s must be finite and positive")

    inverted_counts = []
    volume_ratios = []
    p99_changes = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each substep sample must be a tissue-health mapping")
        inverted = _number(sample, "inverted_tetrahedra")
        p99 = _number(sample, "p99_absolute_volume_change")
        ratio = _number(sample, "minimum_signed_volume_ratio")
        if inverted < 0 or not inverted.is_integer():
            raise ValueError("inverted_tetrahedra must be a nonnegative integer")
        if p99 < 0:
            raise ValueError("p99_absolute_volume_change must be nonnegative")
        inverted_counts.append(int(inverted))
        volume_ratios.append(ratio)
        p99_changes.append(p99)

    return {
        "sampling": "one tissue-health readback after every underlying sim.step()",
        "sampling_complete": True,
        "physics_ticks": len(samples),
        "physics_dt_s": dt,
        "ticks_with_inversion": sum(count > 0 for count in inverted_counts),
        "total_inverted_tetrahedra_across_ticks": sum(inverted_counts),
        "max_inverted_tetrahedra_in_one_tick": max(inverted_counts),
        "minimum_signed_volume_ratio": min(volume_ratios),
        "max_p99_absolute_volume_change": max(p99_changes),
        "all_sampled_ticks_inversion_free": not any(inverted_counts),
    }


class EpisodeSubstepHealth:
    """Aggregate interval summaries without hiding missing physics samples."""

    def __init__(self):
        self.physics_ticks = 0
        self.intervals = 0
        self.sampling_complete = True
        self.ticks_with_inversion = 0
        self.total_inverted_tetrahedra = 0
        self.max_inverted_tetrahedra = 0
        self.minimum_signed_volume_ratio = math.inf
        self.max_p99_absolute_volume_change = 0.0
        self.physics_dt_min_s = math.inf
        self.physics_dt_max_s = 0.0

    def update(self, interval):
        if not isinstance(interval, dict):
            self.sampling_complete = False
            return
        ticks = _number(interval, "physics_ticks")
        inverted_ticks = _number(interval, "ticks_with_inversion")
        total_inverted = _number(interval, "total_inverted_tetrahedra_across_ticks")
        max_inverted = _number(interval, "max_inverted_tetrahedra_in_one_tick")
        minimum_ratio = _number(interval, "minimum_signed_volume_ratio")
        p99 = _number(interval, "max_p99_absolute_volume_change")
        dt = _number(interval, "physics_dt_s")
        if any(value < 0 or not value.is_integer()
               for value in (ticks, inverted_ticks, total_inverted, max_inverted)):
            raise ValueError("Substep counts must be nonnegative integers")
        inconsistent_inversion = ((inverted_ticks == 0) != (total_inverted == 0) or
                                  (inverted_ticks == 0) != (max_inverted == 0) or
                                  interval.get("all_sampled_ticks_inversion_free") != (inverted_ticks == 0))
        if (ticks < 1 or inverted_ticks > ticks or total_inverted < max_inverted or
                dt <= 0 or p99 < 0 or inconsistent_inversion):
            raise ValueError("Inconsistent substep-health interval")

        self.intervals += 1
        self.physics_ticks += int(ticks)
        self.ticks_with_inversion += int(inverted_ticks)
        self.total_inverted_tetrahedra += int(total_inverted)
        self.max_inverted_tetrahedra = max(self.max_inverted_tetrahedra, int(max_inverted))
        self.minimum_signed_volume_ratio = min(self.minimum_signed_volume_ratio, minimum_ratio)
        self.max_p99_absolute_volume_change = max(self.max_p99_absolute_volume_change, p99)
        self.physics_dt_min_s = min(self.physics_dt_min_s, dt)
        self.physics_dt_max_s = max(self.physics_dt_max_s, dt)
        self.sampling_complete &= interval.get("sampling_complete") is True

    def as_dict(self):
        complete = self.sampling_complete and self.intervals > 0 and self.physics_ticks > 0
        return {
            "sampling": "one tissue-health readback after every underlying sim.step() during RL intervals",
            "sampling_complete": complete,
            "intervals": self.intervals,
            "physics_ticks": self.physics_ticks,
            "physics_dt_min_s": None if not self.intervals else self.physics_dt_min_s,
            "physics_dt_max_s": None if not self.intervals else self.physics_dt_max_s,
            "ticks_with_inversion": self.ticks_with_inversion,
            "total_inverted_tetrahedra_across_ticks": self.total_inverted_tetrahedra,
            "max_inverted_tetrahedra_in_one_tick": self.max_inverted_tetrahedra,
            "minimum_signed_volume_ratio": None if not self.intervals else self.minimum_signed_volume_ratio,
            "max_p99_absolute_volume_change": self.max_p99_absolute_volume_change,
            "all_sampled_ticks_inversion_free": complete and self.ticks_with_inversion == 0,
        }


def _number(mapping, key):
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric substep metric: {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite substep metric: {key}")
    return value
