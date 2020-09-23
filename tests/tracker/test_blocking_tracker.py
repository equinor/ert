import unittest

from ert_shared.models import BaseRunModel
from ert_shared.tracker.blocking import BlockingTracker
from ert_shared.tracker.events import DetailedEvent, EndEvent, GeneralEvent


class BlockingTrackerTest(unittest.TestCase):
    def test_event_loop_runs(self):
        brm = BaseRunModel(None, phase_count=1)
        tracker = BlockingTracker(brm, 1, 1)

        idx = 0
        expected = [GeneralEvent, DetailedEvent]
        for event in tracker.track():
            self.assertIsInstance(event, expected[idx], "got unexpected event")

            idx += 1
            if idx == 2:  # End after recieving the detailed event
                break

    def test_end_events_from_finished_model(self):
        brm = BaseRunModel(None, phase_count=0)
        tracker = BlockingTracker(brm, 1, 1)

        events = list(tracker.track())
        for idx, ev_cls in enumerate([GeneralEvent, DetailedEvent, EndEvent]):
            self.assertIsInstance(events[idx], ev_cls)
