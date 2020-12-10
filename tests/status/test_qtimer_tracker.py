import sys
import unittest

from ert_shared.models import BaseRunModel
from ert_shared.tracker.events import DetailedEvent, EndEvent, GeneralEvent
from ert_shared.tracker.qt import QTimerTracker

from unittest.mock import Mock


class QTimerTrackerTest(unittest.TestCase):
    def test_qtimers_are_instantiated_and_setup(self):
        intervals = [2, 3]
        tracker = QTimerTracker(None, Mock, intervals[0], intervals[1], Mock())

        self.assertEqual(2, len(tracker._qtimers), "tracker did not create two timers")
        for idx, interval in enumerate(intervals):
            timer = tracker._qtimers[idx]
            timer.setInterval.assert_called_once_with(interval * 1000)
            timer.timeout.connect.assert_called_once()

    def test_end_events_are_emitted(self):
        event_handler = Mock()
        brm = BaseRunModel(None, phase_count=0)  # a finished model
        tracker = QTimerTracker(brm, Mock, 1, 0, event_handler)

        tracker._general()

        for idx, ev_cls in enumerate([GeneralEvent, DetailedEvent, EndEvent]):
            _, args, _ = event_handler.mock_calls[idx]
            self.assertIsInstance(args[0], ev_cls, "called with unexpected event")

    def test_qtimers_are_stopped_for_finished_model(self):
        brm = BaseRunModel(None, phase_count=0)  # a finished model
        tracker = QTimerTracker(brm, Mock, 1, 0, Mock())

        tracker._general()

        for timer in tracker._qtimers:
            timer.stop.assert_called_once()
