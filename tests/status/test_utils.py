import unittest

from ert_shared.status.utils import (
    format_running_time,
    scale_intervals,
    _calculate_progress,
)


class TrackerUtilsTest(unittest.TestCase):
    def test_format_running_time(self):
        tests = [
            {"seconds": 0, "expected": "Running time: 0 seconds"},
            {"seconds": 1, "expected": "Running time: 1 seconds"},
            {"seconds": 100, "expected": "Running time: 1 minutes 40 seconds"},
            {
                "seconds": 10000,
                "expected": "Running time: 2 hours 46 minutes 40 seconds",
            },  # noqa
            {
                "seconds": 100000,
                "expected": "Running time: 1 days 3 hours 46 minutes 40 seconds",
            },  # noqa
            {
                "seconds": 100000000,
                "expected": "Running time: 1157 days 9 hours 46 minutes 40 seconds",
            },  # noqa
        ]

        for t in tests:
            self.assertEqual(t["expected"], format_running_time(t["seconds"]))

    def test_scale_intervals(self):
        tests = [
            {"reals": 1, "expected_gen": 1, "expected_det": 5},
            {"reals": 100, "expected_gen": 1, "expected_det": 5},
            {"reals": 500, "expected_gen": 5, "expected_det": 15},
            {"reals": 900, "expected_gen": 5, "expected_det": 15},
            {"reals": 1000, "expected_gen": 5, "expected_det": 15},
        ]

        for t in tests:
            actual_gen, actual_det = scale_intervals(t["reals"])
            self.assertEqual(
                t["expected_gen"],
                actual_gen,
                "failed to scale general to {} (was: {}) for {} reals".format(
                    t["expected_gen"], actual_gen, t["reals"]
                ),
            )
            self.assertEqual(
                t["expected_det"],
                actual_det,
                "failed to scale detailed to {} (was: {}) for {} reals".format(
                    t["expected_det"], actual_det, t["reals"]
                ),
            )

    def test_calculate_progress(self):
        tests = [
            {
                "expected": 0.01,
                "phase": 0,
                "phase_count": 1,
                "finished": False,
                "phase_count": 1,
                "total_reals": 100,
                "done_reals": 1,
            },  # noqa
            {
                "expected": 1,
                "phase": 1,
                "phase_count": 1,
                "finished": True,
                "total_reals": 100,
                "done_reals": 100,
            },  # noqa
            {
                "expected": 0.5,
                "phase": 0,
                "phase_count": 2,
                "finished": False,
                "total_reals": 100,
                "done_reals": 100,
            },  # noqa
            {
                "expected": 0,
                "phase": 0,
                "phase_count": 2,
                "finished": False,
                "total_reals": 100,
                "done_reals": 0,
            },
        ]

        for t in tests:
            self.assertEqual(
                t["expected"],
                _calculate_progress(
                    t["finished"],
                    t["phase"],
                    t["phase_count"],
                    t["done_reals"],
                    t["total_reals"],
                ),
            )
