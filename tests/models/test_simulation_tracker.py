import unittest

from ert_shared.models import BaseRunModel, SimulationsTracker


class SimulationTrackerTest(unittest.TestCase):

    def test_track(self):
        brm = BaseRunModel(None)
        tracker = SimulationsTracker(
            update_interval=0, emit_interval=1, model=brm)
        track_iter = tracker.track()

        self.assertEqual(tracker, next(track_iter), "no update")

        brm._phase = 1  # model now done, track should stop
        self.assertEqual(tracker, next(track_iter), "no final update")

        with self.assertRaises(StopIteration,
                               msg="tracker did not stop"):
            next(track_iter)

    def test_format_running_time(self):
        tests = [
            {"seconds": 0, "expected": "Running time: 0 seconds"},
            {"seconds": 1, "expected": "Running time: 1 seconds"},
            {"seconds": 100, "expected": "Running time: 1 minutes 40 seconds"},
            {"seconds": 10000, "expected": "Running time: 2 hours 46 minutes 40 seconds"}, # noqa
            {"seconds": 100000, "expected": "Running time: 1 days 3 hours 46 minutes 40 seconds"},  # noqa
            {"seconds": 100000000, "expected": "Running time: 1157 days 9 hours 46 minutes 40 seconds"},  # noqa
        ]

        for t in tests:
            self.assertEqual(
                t["expected"],
                SimulationsTracker.format_running_time(t["seconds"])
            )
