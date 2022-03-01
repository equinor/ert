import unittest

from ert_shared.status.utils import (
    _calculate_progress,
    format_running_time,
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
