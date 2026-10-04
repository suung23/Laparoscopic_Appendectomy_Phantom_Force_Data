"""Conservative contact supervisor, using measured joint reactions.

The locally tared mount residual includes inertial/payload-model error. It is
not a calibrated tissue force. The jaw effort/lever ratio is likewise a proxy.
Thresholds are simulator engineering settings, not surgical safety limits.
"""
import numpy as np


class ContactGuard:
    def __init__(self, baseline, noise_n, stop_n=.15, jaw_effort_nm=.0003):
        self.baseline = np.asarray(baseline, float).copy()
        if self.baseline.shape != (6,) or not np.isfinite(self.baseline).all():
            raise ValueError('Finite six-axis baseline required')
        if not np.isfinite([noise_n, stop_n, jaw_effort_nm]).all() or noise_n < 0 or min(stop_n, jaw_effort_nm) <= 0:
            raise ValueError('Invalid thresholds')
        self.threshold = max(stop_n, 8*noise_n)
        self.jaw_limit = jaw_effort_nm
        self.filtered = np.zeros(6)
        self.latched = False
        self.samples = 0
        self.last = {}
        self.pinch_latched = False
        self.blocked_angle = None

    def update(self, wrench, jaw_efforts):
        w = np.asarray(wrench, float); j = np.asarray(jaw_efforts, float)
        if w.shape != (6,) or j.shape != (2,) or not np.isfinite(w).all() or not np.isfinite(j).all():
            raise RuntimeError('Contact sensor unavailable/nonfinite')
        residual = w-self.baseline
        self.filtered += .2*(residual-self.filtered)
        force = float(np.linalg.norm(self.filtered[:3]))
        # Latch a stop: no automatic re-approach/chatter into the same surface.
        self.latched |= force > self.threshold
        self.pinch_latched |= np.max(np.abs(j)) >= self.jaw_limit
        self.samples += 1
        self.last = dict(mount_reaction=w.tolist(), mount_residual=self.filtered.tolist(),
            residual_force_n=force, stop_threshold_n=self.threshold, motion_latched=bool(self.latched),
            jaw_effort_nm=j.tolist(), jaw_load_proxy_n=float(np.max(np.abs(j))/.015),
            jaw_limited=bool(self.pinch_latched),
            force_source='PhysX joint reactions; local static tare, not calibrated tissue force')
        return self.last

    def jaw_target(self, requested_deg, previous_deg, measured_deg, dt):
        # Opening always remains available. Closing has bounded speed and no
        # position integrator windup beyond the measured jaw under resistance.
        if not np.isfinite([requested_deg, previous_deg, measured_deg, dt]).all() or dt <= 0:
            raise RuntimeError('Invalid jaw feedback')
        target = previous_deg + float(np.clip(requested_deg-previous_deg, -30*dt, 90*dt))
        if requested_deg > previous_deg:
            self.pinch_latched = False; self.blocked_angle = None
        if requested_deg < previous_deg:
            target = max(target, measured_deg-1.)
            if self.pinch_latched:
                if self.blocked_angle is None:self.blocked_angle = measured_deg+.25
                target = max(target, self.blocked_angle)
        return float(np.clip(target, 0, 74))
