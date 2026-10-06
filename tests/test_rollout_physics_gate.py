import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac"))
from vla_rollout import Episode, physics_health_gate


class RolloutPhysicsGateTests(unittest.TestCase):
    def test_health_gate_passes_only_positive_finite_tissue(self):
        self.assertTrue(physics_health_gate([1, 1], [1, .8], [0, 0], [0, 0]))
        self.assertFalse(physics_health_gate([1, 0], [1, .8], [0, 0], [0, 0]))
        self.assertFalse(physics_health_gate([1], [1, -.2], [0], [0]))
        self.assertFalse(physics_health_gate([1], [1], [1], [0]))
        self.assertFalse(physics_health_gate([1], [1], [0], [1]))
        self.assertFalse(physics_health_gate([1], [np.nan], [0], [0]))

    def test_absent_backend_telemetry_is_unavailable(self):
        self.assertIsNone(physics_health_gate([np.nan], [np.nan],
                                              [np.nan], [np.nan]))

    def test_invalid_physics_invalidates_episode_task_success(self):
        episode = Episode(0, keep_images=False)
        episode.add({
            "t": 0.0,
            "T_world_rig": np.eye(4, dtype=np.float32),
            "task": {
                "physics_state_finite": True,
                "tissue_min_signed_volume_ratio": 1.0,
                "tissue_inverted_tets": 1,
                "tissue_nonfinite_tets": 0,
            },
        })
        episode.physics_failure = "1 inverted tetrahedra"
        score = episode.score()
        self.assertFalse(score["physics_health_pass"])
        self.assertFalse(score["success"])
        self.assertFalse(score["grasp_lift_success"])
        self.assertEqual(score["physics_failure"], "1 inverted tetrahedra")


if __name__ == "__main__":
    unittest.main()
