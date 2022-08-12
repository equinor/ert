import unittest

from ert.shared.status.utils import (
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
