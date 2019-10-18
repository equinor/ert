# -*- coding: utf-8 -*-
import sys
import unittest
from ert_shared.models import BaseRunModel, SimulationsTracker
from ert_shared.cli.monitor import Monitor


if sys.version_info >= (3, 5):
    from io import StringIO
else:
    from io import BytesIO as StringIO

class MonitorTest(unittest.TestCase):

    def test_color_always(self):
        out = StringIO()  # not atty, so coloring is automatically disabled
        monitor = Monitor(out=out, color_always=True)

        self.assertEqual("\x1b[38;2;255;0;0mFoo\x1b[0m",
                         monitor._colorize("Foo", fg=(255, 0, 0)))

    def test_legends(self):
        sim_tracker = SimulationsTracker()
        done_state = sim_tracker.getStates()[0]  # first is the Finished state
        done_state.total_count = 100
        done_state.count = 10
        monitor = Monitor(out=StringIO())

        legends = monitor._get_legends(sim_tracker)

        self.assertEqual("Finished       10/100", legends[done_state])

    def test_result_success(self):
        out = StringIO()
        monitor = Monitor(out=out)

        monitor._print_result(False, None)

        self.assertEqual("Simulations completed.\n", out.getvalue())

    def test_result_failure(self):
        out = StringIO()
        monitor = Monitor(out=out)

        monitor._print_result(True, "fail")

        self.assertEqual(
            "Simulations failed with the following error: fail\n",
            out.getvalue()
        )

    def test_print_progress(self):
        out = StringIO()
        sim_tracker = SimulationsTracker(model=BaseRunModel(None))
        monitor = Monitor(out=out)
        sim_tracker._update()
        sim_tracker.getStates()[0].count = 1

        monitor._print_progress(sim_tracker)

        self.assertEqual(
            """\r
    --> Starting...

    1/1 |██████████████████████████████| 100% Running time: 0 seconds

    Finished          1/1
    Failed            0/1
    Running           0/1
    Unknown           0/1
    Pending           0/1
    Waiting           0/1

""", out.getvalue())
