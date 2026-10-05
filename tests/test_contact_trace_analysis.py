import unittest

from ct_appendectomy.analyze_contact_trial import summarize_trace


def tick(left_effort=0.0, right_effort=0.0, inverted=0, ratio=0.9, limited=False):
    return {
        "contact": {
            "left": {"jaw_effort_nm": [left_effort, 0.0], "residual_force_n": .02,
                     "jaw_limited": limited},
            "right": {"jaw_effort_nm": [right_effort, 0.0], "residual_force_n": .03,
                      "jaw_limited": False},
        },
        "tissue_health": {"minimum_signed_volume_ratio": ratio,
                          "inverted_tetrahedra": inverted},
    }


class ContactTraceAnalysisTests(unittest.TestCase):
    def test_reports_force_gate_and_inversion_order_with_trace_time(self):
        trace = [tick(), tick(.00031, limited=True, ratio=.7), tick(.0004, ratio=.5),
                 tick(.0005, inverted=1, ratio=-.1)]
        summary = summarize_trace({"physics_hz": 480, "physics_ticks_checked": 4,
                                   "status": "failed", "failure": "tet_inversion"}, trace)
        self.assertEqual(summary["first_any_jaw_threshold_crossing_tick"], 1)
        self.assertEqual(summary["first_inversion_tick"], 3)
        self.assertTrue(summary["jaw_effort_trip_precedes_or_matches_first_inversion"])
        self.assertAlmostEqual(summary["arms"]["left"]["first_threshold_crossing_trace_time_s"], 1/480)
        self.assertEqual(summary["reported_failure"], "tet_inversion")

    def test_rejects_tick_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "tick count"):
            summarize_trace({"physics_hz": 480, "physics_ticks_checked": 2}, [tick()])


if __name__ == "__main__":
    unittest.main()
