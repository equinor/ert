# -*- coding: utf-8 -*-
import sys
import unittest
from ert_shared.tracker.events import GeneralEvent
from ert_shared.tracker.state import SimulationStateStatus
from ert_shared.cli.monitor import Monitor


from io import StringIO


class MonitorTest(unittest.TestCase):

    def test_color_always(self):
        out = StringIO()  # not atty, so coloring is automatically disabled
        monitor = Monitor(out=out, color_always=True)

        self.assertEqual("\x1b[38;2;255;0;0mFoo\x1b[0m",
                         monitor._colorize("Foo", fg=(255, 0, 0)))

    def test_legends(self):
        done_state = SimulationStateStatus("Finished", None, None)
        done_state.count = 10
        done_state.total_count = 100
        monitor = Monitor(out=StringIO())

        legends = monitor._get_legends([done_state])

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
        monitor = Monitor(out=out)
        states = [
            SimulationStateStatus("Finished", None, None),
            SimulationStateStatus("Waiting", None, None),
        ]
        states[0].count = 10
        states[0].total_count = 100
        general_event = GeneralEvent("Test Phase", 0, 2, 0.5, False, states, 10)

        monitor._print_progress(general_event)

        self.assertEqual(
            """\r
    --> Test Phase

    1/2 |███████████████               | 50% Running time: 10 seconds

    Finished       10/100
    Waiting           0/1
""", out.getvalue())
