"""Aggregate control-step tissue-safety metrics for RL episode records."""
from __future__ import annotations

import math


class EpisodePhysicsMetrics:
    """Track worst observed deformation and mesh-health values per episode.

    Samples are control-step readbacks, not every PhysX substep. This class
    deliberately does not infer tissue force from deformation or jaw motion.
    """

    def __init__(self):
        self.samples = 0
        self.max_pin_drift_m = 0.0
        self.max_tracking_error_m = 0.0
        self.max_node_displacement_m = 0.0
        self.minimum_signed_volume_ratio = math.inf
        self.max_p99_absolute_volume_change = 0.0
        self.max_inverted_tetrahedra = 0
        self.inversion_steps = 0

    @staticmethod
    def _finite(info, key):
        try:
            value = float(info[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing numeric physics metric: {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite physics metric: {key}")
        return value

    def update(self, info):
        health = info.get("tissue_health")
        if not isinstance(health, dict):
            raise ValueError("Episode info is missing tissue_health")
        pin = self._finite(info, "pin_drift_m")
        tracking = self._finite(info, "tracking_error_m")
        displacement = self._finite(info, "maximum_displacement_m")
        volume_ratio = self._finite(health, "minimum_signed_volume_ratio")
        p99 = self._finite(health, "p99_absolute_volume_change")
        inverted_value = self._finite(health, "inverted_tetrahedra")
        if inverted_value < 0 or not inverted_value.is_integer():
            raise ValueError("inverted_tetrahedra must be a nonnegative integer")
        inverted = int(inverted_value)

        self.samples += 1
        self.max_pin_drift_m = max(self.max_pin_drift_m, pin)
        self.max_tracking_error_m = max(self.max_tracking_error_m, tracking)
        self.max_node_displacement_m = max(self.max_node_displacement_m, displacement)
        self.minimum_signed_volume_ratio = min(self.minimum_signed_volume_ratio, volume_ratio)
        self.max_p99_absolute_volume_change = max(self.max_p99_absolute_volume_change, p99)
        self.max_inverted_tetrahedra = max(self.max_inverted_tetrahedra, inverted)
        self.inversion_steps += int(inverted > 0)

    def as_dict(self):
        return {
            "sampling": "one readback per agent control step; not every physics substep",
            "samples": self.samples,
            "max_pin_drift_m": self.max_pin_drift_m,
            "max_tracking_error_m": self.max_tracking_error_m,
            "max_node_displacement_m": self.max_node_displacement_m,
            "minimum_signed_volume_ratio": None if not self.samples else self.minimum_signed_volume_ratio,
            "max_p99_absolute_volume_change": self.max_p99_absolute_volume_change,
            "max_inverted_tetrahedra": self.max_inverted_tetrahedra,
            "inversion_steps": self.inversion_steps,
            "force_readback_available": False,
            "force_interpretation": "not measured by this 2-actuator contact-lift RL fixture",
        }
