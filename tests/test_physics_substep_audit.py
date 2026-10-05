import unittest
import sys
from pathlib import Path

from ct_appendectomy.physics_substep_audit import EpisodeSubstepHealth, summarize_interval
from ct_appendectomy.run_rosota_rl import evaluate_policy, summarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from rosota_ct_env import RosotaCTEnv


def health(inverted, ratio, p99):
    return {
        "inverted_tetrahedra": inverted,
        "minimum_signed_volume_ratio": ratio,
        "p99_absolute_volume_change": p99,
    }


class PhysicsSubstepAuditTests(unittest.TestCase):
    def test_transient_inversion_is_retained_when_endpoint_recovers(self):
        interval = summarize_interval([
            health(0, .91, .08),
            health(3, -.12, .31),
            health(0, .89, .11),
        ], 1 / 1920)
        self.assertEqual(interval["physics_ticks"], 3)
        self.assertEqual(interval["ticks_with_inversion"], 1)
        self.assertEqual(interval["max_inverted_tetrahedra_in_one_tick"], 3)
        self.assertFalse(interval["all_sampled_ticks_inversion_free"])

    def test_episode_aggregates_every_interval_and_exposes_safe_gate(self):
        audit = EpisodeSubstepHealth()
        audit.update(summarize_interval([health(0, .93, .06)] * 16, 1 / 480))
        audit.update(summarize_interval([health(0, .88, .14)] * 16, 1 / 1920))
        result = audit.as_dict()
        self.assertTrue(result["sampling_complete"])
        self.assertEqual(result["physics_ticks"], 32)
        self.assertEqual(result["ticks_with_inversion"], 0)
        self.assertTrue(result["all_sampled_ticks_inversion_free"])
        self.assertAlmostEqual(result["minimum_signed_volume_ratio"], .88)
        self.assertAlmostEqual(result["physics_dt_min_s"], 1 / 1920)

    def test_missing_interval_fails_closed(self):
        audit = EpisodeSubstepHealth()
        audit.update(None)
        result = audit.as_dict()
        self.assertFalse(result["sampling_complete"])
        self.assertFalse(result["all_sampled_ticks_inversion_free"])

    def test_rejects_inconsistent_inversion_counts(self):
        interval = summarize_interval([health(0, 1.0, 0.0)], 1 / 480)
        interval["ticks_with_inversion"] = 1
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            EpisodeSubstepHealth().update(interval)

    def test_rejects_nonfinite_tissue_health(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            summarize_interval([health(0, float("nan"), .0)], 1 / 480)

    def test_evaluation_report_carries_all_solver_tick_counts(self):
        class ScriptedEnv:
            def __init__(self):
                self.steps = 0

            def reset(self, *, seed):
                self.steps = 0
                return object(), {}

            def scripted_waypoint_action(self):
                return [0.0]

            def step(self, action):
                self.steps += 1
                interval = summarize_interval([health(0, .9, .1)] * 16, 1 / 480)
                info = {
                    "waypoint_distance_m": [.001, .002],
                    "waypoint_is_success": True,
                    "physics_substep_health": interval,
                }
                return object(), 1.0, self.steps == 2, False, info

        result = evaluate_policy(ScriptedEnv(), None, "scripted", seeds=[71])[0]
        audit = result["physics_substep_health"]
        self.assertTrue(audit["sampling_complete"])
        self.assertEqual(audit["physics_ticks"], 32)
        self.assertTrue(audit["all_sampled_ticks_inversion_free"])
        summary = summarize([result])["physics_substep_safety"]
        self.assertEqual(summary["physics_ticks"], 32)
        self.assertTrue(summary["sampling_complete"])
        self.assertTrue(summary["all_sampled_ticks_inversion_free"])

    def test_physics_step_stops_remaining_substeps_after_failed_health_sample(self):
        class FakeSimulation:
            def __init__(self):
                self.steps = 0

            def step(self, *, render):
                self.steps += 1

        env = object.__new__(RosotaCTEnv)
        env.physics_substeps = 4
        env.sim = FakeSimulation()
        samples = []

        def sample():
            samples.append(env.sim.steps)
            return len(samples) < 2

        env._physics_step(sample)
        self.assertEqual(samples, [1, 2])
        self.assertEqual(env.sim.steps, 2)


if __name__ == "__main__":
    unittest.main()
