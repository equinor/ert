import unittest

from ert_shared.tracker.base import BaseTracker
from res.job_queue import JobStatusType

from unittest.mock import Mock


class BaseTrackerTest(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.tracker = BaseTracker(self.model, None)

    def test_general_event_generation(self):
        self.model.getPhaseName.return_value = "Test Phase"
        self.model.currentPhase.return_value = 1
        self.model.phaseCount.return_value = 2
        self.model.getQueueSize.return_value = 100
        self.model.isFinished.return_value = False
        self.model.isQueueRunning.return_value = True
        self.model.isIndeterminate.return_value = True
        self.model.start_time.return_value = 100
        self.model.stop_time.return_value = 200
        self.model.getQueueStatus.return_value = {JobStatusType.JOB_QUEUE_DONE: 50}
        self.model.get_runtime.return_value = 100

        general_event = self.tracker._general_event()

        self.assertEqual("Test Phase", general_event.phase_name)
        self.assertEqual(1, general_event.current_phase)
        self.assertEqual(2, general_event.total_phases)
        self.assertEqual(0.75, general_event.progress)
        self.assertEqual(True, general_event.indeterminate)
        self.assertEqual(100, general_event.runtime)

    def test_detailed_event_generation(self):
        self.model.getDetailedProgress.return_value = {}, -1

        detailed_event = self.tracker._detailed_event()

        self.assertEqual({}, detailed_event.details)
        self.assertEqual(-1, detailed_event.iteration)
