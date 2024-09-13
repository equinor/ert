from datetime import datetime
from io import StringIO

from ert.cli.monitor import Monitor
from ert.ensemble_evaluator.event import SnapshotUpdateEvent
from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
    RealizationSnapshot,
)
from ert.ensemble_evaluator.state import (
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_RUNNING,
    REALIZATION_STATE_WAITING,
)


def test_color_always():
    out = StringIO()  # not a tty, so coloring is automatically disabled
    monitor = Monitor(out=out, color_always=True)

    assert monitor._colorize("Foo", color=(255, 0, 0)) == "\x1b[38;2;255;0;0mFoo\x1b[0m"


def test_legends():
    monitor = Monitor(out=StringIO())
    snapshot = EnsembleSnapshot()
    snapshot._ensemble_state = ""
    for i in range(0, 100):
        status = REALIZATION_STATE_FINISHED if i < 10 else REALIZATION_STATE_RUNNING
        snapshot.add_realization(
            str(i), RealizationSnapshot(status=status, active=True)
        )
    monitor._snapshots[0] = snapshot
    legends = monitor._get_legends()

    assert (
        legends
        == """    Waiting         0/100
    Pending         0/100
    Running        90/100
    Failed          0/100
    Finished       10/100
    Unknown         0/100
"""
    )


def test_result_success():
    out = StringIO()
    monitor = Monitor(out=out)

    monitor._print_result(False, None)

    assert out.getvalue() == "Experiment completed.\n"


def test_result_failure():
    out = StringIO()
    monitor = Monitor(out=out)

    monitor._print_result(True, "fail")

    assert out.getvalue() == "Experiment failed with the following error: fail\n"


def test_print_progress():
    out = StringIO()
    monitor = Monitor(out=out)
    snapshot = EnsembleSnapshot()
    snapshot._ensemble_state = ""
    for i in range(0, 100):
        status = REALIZATION_STATE_FINISHED if i < 50 else REALIZATION_STATE_WAITING
        snapshot.add_realization(
            str(i), RealizationSnapshot(status=status, active=True)
        )
    monitor._snapshots[0] = snapshot
    monitor._start_time = datetime.now()
    general_event = SnapshotUpdateEvent(
        iteration_label="Test Phase",
        current_iteration=0,
        total_iterations=2,
        progress=0.5,
        realization_count=100,
        status_count={"Finished": 50, "Waiting": 50},
        iteration=0,
    )

    monitor._print_progress(general_event)

    # For some reason, `tqdm` adds an extra line containing a progress-bar,
    # even though this test only calls it once.
    # I suspect this has something to do with the way `tqdm` does refresh,
    # but do not know how to fix it.
    # Seems not be a an issue when used normally.
    expected = """    --> Test Phase


    |                                                                                      |   0% it
    1/2 |##############################5                              |  50% Running time: 0 seconds

    Waiting        50/100
    Pending         0/100
    Running         0/100
    Failed          0/100
    Finished       50/100
    Unknown         0/100

"""

    assert out.getvalue().replace("\r", "\n") == expected
