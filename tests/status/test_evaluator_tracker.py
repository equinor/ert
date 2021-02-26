from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
import logging
from ert_shared.models.base_run_model import BaseRunModel
from ert_shared.status.tracker.evaluator import EvaluatorTracker
from unittest.mock import MagicMock, patch
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    SnapshotBuilder,
    Step,
)

from ert_shared.status.entity import state


class MockCloudEvent(dict):
    def __init__(self, dict_, data):
        self.__dict__ = dict_
        self.data = data

    def __getitem__(self, key):
        return self.__dict__[key]


def mock_ee_monitor(*args):
    reals_ids = ["0", "1"]
    snapshot = (
        SnapshotBuilder()
        .add_stage(stage_id="0", status=state.STAGE_STATE_RUNNING)
        .add_step(stage_id="0", step_id="0", status=state.STEP_STATE_START)
        .add_job(
            stage_id="0",
            step_id="0",
            job_id="0",
            name="job0",
            data={},
            status=state.JOB_STATE_RUNNING,
        )
        .build(reals_ids, state.REALIZATION_STATE_UNKNOWN)
    )

    update = PartialSnapshot(snapshot)
    update.update_step("0", "0", "0", step=Step(status=state.STEP_STATE_SUCCESS))
    update.update_step("1", "0", "0", step=Step(status=state.STEP_STATE_SUCCESS))

    events = [
        MockCloudEvent(
            {"source": "/", "time": None, "type": ids.EVTYPE_EE_SNAPSHOT},
            {**(snapshot.to_dict()), "iter": 0},
        ),
        MockCloudEvent(
            {"source": "/", "time": None, "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
            {**update.to_dict(), "iter": 0},
        ),
        MockCloudEvent(
            {"source": "/", "time": None, "type": ids.EVTYPE_EE_TERMINATED}, {"iter": 0}
        ),
    ]

    def _track():
        while True:
            try:
                yield events.pop(0)
            except IndexError:
                return

    return MagicMock(track=MagicMock(side_effect=_track))


def test_tracking(caplog):
    caplog.set_level(logging.DEBUG, logger="ert_shared.ensemble_evaluator")
    brm = BaseRunModel(None, phase_count=1)

    with patch("ert_shared.status.tracker.evaluator.create_ee_monitor") as mock_ee:
        mock_ee.return_value.__enter__.return_value = mock_ee_monitor()
        tracker = EvaluatorTracker(brm, "host", "port", 1, 1)
        tracker_gen = tracker.track()

        event = next(tracker_gen)
        assert isinstance(event, FullSnapshotEvent)

        event = next(tracker_gen)
        assert isinstance(event, SnapshotUpdateEvent)

        # this is setting the model to finished
        brm._phase = 1

        event = next(tracker_gen)
        assert isinstance(event, EndEvent)
