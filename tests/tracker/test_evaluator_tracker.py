import time
import logging
from res.job_queue import JobStatusType
from ert_shared.models.base_run_model import BaseRunModel
from ert_shared.tracker.evaluator import EvaluatorTracker
from unittest.mock import MagicMock, patch
from ert_shared.tracker.state import SimulationStateStatus
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    SnapshotBuilder,
)


def generate_some_states():
    return [
        SimulationStateStatus(
            "Unknown",
            JobStatusType.JOB_QUEUE_UNKNOWN,
            SimulationStateStatus.COLOR_RUNNING,
        ),
        SimulationStateStatus(
            "Running",
            JobStatusType.JOB_QUEUE_RUNNING,
            SimulationStateStatus.COLOR_RUNNING,
        ),
        SimulationStateStatus(
            "Waiting",
            JobStatusType.JOB_QUEUE_WAITING,
            SimulationStateStatus.COLOR_WAITING,
        ),
        SimulationStateStatus(
            "Finished",
            JobStatusType.JOB_QUEUE_SUCCESS,
            SimulationStateStatus.COLOR_FINISHED,
        ),
    ]


class MockCloudEvent(dict):
    def __init__(self, dict_, data):
        self.__dict__ = dict_
        self.data = data

    def __getitem__(self, key):
        return self.__dict__[key]


def mock_ee_monitor(*args):
    reals_ids = ["0", "1"]
    events = [
        MockCloudEvent(
            {"type": ids.EVTYPE_EE_SNAPSHOT},
            SnapshotBuilder()
            .add_stage(stage_id="0", status="Running", queue_state="JOB_QUEUE_RUNNING")
            .add_step(stage_id="0", step_id="0", status="Unknown")
            .add_job(
                stage_id="0",
                step_id="0",
                job_id="0",
                name="job0",
                data={},
                status="Running",
            )
            .add_metadata("iter", 0)
            .build(reals_ids, "Unknown")
            .to_dict(),
        ),
        MockCloudEvent(
            {"type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
            SnapshotBuilder()
            .add_stage(stage_id="0", status="Finished", queue_state="JOB_QUEUE_SUCCESS")
            .add_step(stage_id="0", step_id="0", status="Finished")
            .add_metadata("iter", 0)
            .build(reals_ids, "Unknown")
            .to_dict(),
        ),
        MockCloudEvent({"type": ids.EVTYPE_EE_TERMINATED}, {}),
    ]

    def _track():
        while True:
            try:
                yield events.pop(0)
            except IndexError:
                return

    return MagicMock(track=MagicMock(side_effect=_track))


def test_general_event(caplog):
    caplog.set_level(logging.DEBUG, logger="ert_shared.ensemble_evaluator")
    brm = BaseRunModel(None, phase_count=0)

    with patch("ert_shared.tracker.evaluator.create_ee_monitor") as mock_ee:
        mock_ee.return_value = mock_ee_monitor()
        tracker = EvaluatorTracker(brm, ("", ""), generate_some_states())
        general_event = None
        while not tracker.is_finished():
            general_event = tracker.general_event()
            time.sleep(0.2)
        assert general_event is not None, "no general event"
        for state in general_event.sim_states:
            if state.name == "Finished":
                assert state.count == 2
            else:
                assert state.count == 0, f"{state.name} had non-zero count"


def test_detailed_event(caplog):
    caplog.set_level(logging.DEBUG, logger="ert_shared.ensemble_evaluator")
    brm = BaseRunModel(None, phase_count=0)

    with patch("ert_shared.tracker.evaluator.create_ee_monitor") as mock_ee:
        mock_ee.return_value = mock_ee_monitor()
        tracker = EvaluatorTracker(brm, ("", ""), generate_some_states())

        # allow tracker to complete
        while not tracker.is_finished():
            tracker.general_event()
        detailed_event = tracker.detailed_event()
        assert detailed_event.iteration == 0
        for iter_num, iter_ in detailed_event.details.items():
            for real_num, real in iter_.items():
                job_data = real[0][0].dump_data()
                assert "Running" == job_data["status"]
