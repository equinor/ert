import time
from unittest.mock import patch


from ert_gui.simulation.run_dialog import RunDialog
from ert_shared.status.entity.event import EndEvent, FullSnapshotEvent
from qtpy.QtCore import Qt


class MockTracker:
    def __init__(self, events) -> None:
        self._events = events

    def track(self):
        for event in self._events:
            yield event
            time.sleep(0.1)

    def reset(self):
        pass


def test_success(runmodel, active_realizations, qtbot):
    widget = RunDialog(
        "poly.ert", runmodel, {"active_realizations": active_realizations}
    )
    widget.has_failed_realizations = lambda: False
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.create_tracker") as mock_tracker_factory:
        mock_tracker_factory.return_value = MockTracker(
            [EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    qtbot.waitForWindowShown(widget)
    qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100)
    assert widget.done_button.isVisible()
    assert widget.done_button.text() == "Done"


def test_large_snapshot(runmodel, active_realizations, large_snapshot, qtbot):
    widget = RunDialog(
        "poly.ert", runmodel, {"active_realizations": active_realizations}
    )
    widget.has_failed_realizations = lambda: False
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.create_tracker") as mock_tracker_factory:
        iter_0 = FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=0,
            indeterminate=False,
        )
        iter_1 = FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=1,
            indeterminate=False,
        )
        mock_tracker_factory.return_value = MockTracker(
            [iter_0, iter_1, EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    qtbot.waitForWindowShown(widget)
    qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100, timeout=5000)
    qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: widget._tab_widget.count() == 2, timeout=5000)
