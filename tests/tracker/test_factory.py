import sys
import unittest

from ert_shared.tracker.blocking import BlockingTracker
from ert_shared.tracker.factory import create_tracker
from ert_shared.tracker.qt import QTimerTracker

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


class TrackerFactoryTest(unittest.TestCase):

    def test_create_trackers(self):
        q_tracker = create_tracker(None, qtimer_cls=Mock(),
                                   event_handler=Mock())
        self.assertIsInstance(q_tracker, QTimerTracker,
                              "failed to create QTimerTracker")

        blocking_tracker = create_tracker(None)
        self.assertIsInstance(blocking_tracker, BlockingTracker,
                              "failed to create BlockingTracker")
