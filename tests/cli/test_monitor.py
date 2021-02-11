from datetime import datetime
from ert_shared.status.entity.event import _UpdateEvent
from ert_shared.status.entity.state import (
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_RUNNING,
    REALIZATION_STATE_WAITING,
)
from ert_shared.ensemble_evaluator.entity.snapshot import (
    Snapshot,
    SnapshotDict,
    ForwardModel,
    Realization,
)
import unittest

from ert_shared.cli.monitor import Monitor


from io import StringIO


class MonitorTest(unittest.TestCase):
    def test_color_always(self):
        out = StringIO()  # not atty, so coloring is automatically disabled
        monitor = Monitor(out=out, color_always=True)

        self.assertEqual(
            "\x1b[38;2;255;0;0mFoo\x1b[0m", monitor._colorize("Foo", fg=(255, 0, 0))
        )

    def test_legends(self):
        monitor = Monitor(out=StringIO())
        sd = SnapshotDict(status="", forward_model=ForwardModel(step_definitions={}))
        for i in range(0, 100):
            status = REALIZATION_STATE_FINISHED if i < 10 else REALIZATION_STATE_RUNNING
            sd.reals[i] = Realization(status=status, active=True)
        monitor._snapshot = Snapshot(sd.dict())
        legends = monitor._get_legends()

        self.assertEqual(
            """    Waiting         0/100
    Pending         0/100
    Running        90/100
    Failed          0/100
    Finished       10/100
    Unknown         0/100
""",
            legends,
        )

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
            "Simulations failed with the following error: fail\n", out.getvalue()
        )

    def test_print_progress(self):
        out = StringIO()
        monitor = Monitor(out=out)
        sd = SnapshotDict(status="", forward_model=ForwardModel(step_definitions={}))
        for i in range(0, 100):
            status = REALIZATION_STATE_FINISHED if i < 50 else REALIZATION_STATE_WAITING
            sd.reals[i] = Realization(status=status, active=True)
        monitor._snapshot = Snapshot(sd.dict())
        monitor._start_time = datetime.now()
        general_event = _UpdateEvent(
            phase_name="Test Phase",
            current_phase=0,
            total_phases=2,
            progress=0.5,
            indeterminate=False,
            iteration=0,
        )

        monitor._print_progress(general_event)

        self.assertEqual(
            """\r
    --> Test Phase

    1/2 |███████████████               | 50% Running time: 0 seconds

    Waiting        50/100
    Pending         0/100
    Running         0/100
    Failed          0/100
    Finished       50/100
    Unknown         0/100
""",
            out.getvalue(),
        )
