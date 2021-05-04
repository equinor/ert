from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import ert_shared.ensemble_evaluator.entity.identifiers as ids
import pytest
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    SnapshotBuilder,
    Step,
)
from ert_shared.models.base_run_model import BaseRunModel
from ert_shared.status.entity import state
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert_shared.status.tracker.evaluator import EvaluatorTracker


def build_snapshot(real_list: List[str] = ["0"]):
    return (
        SnapshotBuilder()
        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
        .build(real_list, state.REALIZATION_STATE_UNKNOWN)
    )


def build_partial(real_list: List[str] = ["0"]):
    partial = PartialSnapshot(build_snapshot(real_list))
    return partial


@pytest.fixture
def make_mock_ee_monitor():
    def _mock_ee_monitor(events):
        def _track():
            while True:
                try:
                    event = events.pop(0)
                    yield event
                except IndexError:
                    return

        return MagicMock(track=MagicMock(side_effect=_track))

    return _mock_ee_monitor


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "monitor_events,expected_types",
    [
        pytest.param(
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={**(build_snapshot().to_dict()), "iter": 0},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={**build_partial().to_dict(), "iter": 0},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_TERMINATED}, data={"iter": 0}
                ),
            ],
            [
                FullSnapshotEvent,
                SnapshotUpdateEvent,
                EndEvent,
            ],
            id="single_successful_ensemble",
        ),
        pytest.param(
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={**(build_snapshot().to_dict()), "iter": 0},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={"status": state.ENSEMBLE_STATE_CANCELLED, "iter": 0},
                ),
            ],
            [
                FullSnapshotEvent,
                SnapshotUpdateEvent,
                EndEvent,
            ],
            id="cancelled_ensemble",
        ),
        pytest.param(
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={**(build_snapshot().to_dict()), "iter": 0},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={"status": state.ENSEMBLE_STATE_FAILED, "iter": 0},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_TERMINATED}, data={"iter": 0}
                ),
            ],
            [
                FullSnapshotEvent,
                SnapshotUpdateEvent,
                EndEvent,
            ],
            id="failing_ensemble",
        ),
        pytest.param(
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={**(build_snapshot().to_dict()), "iter": 0},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={**build_partial().to_dict(), "iter": 0},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={"status": state.ENSEMBLE_STATE_STOPPED, "iter": 0},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_TERMINATED}, data={"iter": 0}
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={**(build_snapshot().to_dict()), "iter": 1},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={**build_partial().to_dict(), "iter": 1},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={"status": state.ENSEMBLE_STATE_STOPPED, "iter": 1},
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_TERMINATED}, data={"iter": 1}
                ),
            ],
            [
                FullSnapshotEvent,
                SnapshotUpdateEvent,
                FullSnapshotEvent,
                SnapshotUpdateEvent,
                EndEvent,
            ],
            id="multiple_ensembles",
        ),
    ],
)
def test_batching(
    monitor_events: List[CloudEvent],
    expected_types: List[Any],
    make_mock_ee_monitor,
):
    """Test that monitor_events, when passed through this tracker, produces the
    expected event types. The contents of the events are ignored."""
    brm = BaseRunModel(None, phase_count=1)
    with patch("ert_shared.status.tracker.evaluator.create_ee_monitor") as mock_ee:
        mock_ee.return_value.__enter__.return_value = make_mock_ee_monitor(
            monitor_events
        )
        tracker = EvaluatorTracker(
            brm, "host", "port", 0.1, 1, next_ensemble_evaluator_wait_time=0.1
        )
        tracker_gen = tracker.track()
        for i, type_ in enumerate(expected_types):
            if type_ == EndEvent:
                brm._phase = 1
            event = next(tracker_gen)
            assert isinstance(event, type_), f"unexpected type at index {i}"


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "monitor_events,brm_mutations,expected_progress",
    [
        pytest.param(
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={
                        **(build_snapshot(["0", "1"]).to_dict()),
                        "iter": 0,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={
                        **(
                            build_partial(["0", "1"])
                            .update_step(
                                "0", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .to_dict()
                        ),
                        "iter": 0,
                    },
                ),
            ],
            [("_phase", 0), ("_phase_count", 1)],
            0.5,
            id="ensemble_experiment_50",
        ),
        pytest.param(
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={
                        **(build_snapshot(["0", "1"]).to_dict()),
                        "iter": 0,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={
                        **(
                            build_partial(["0", "1"])
                            .update_step(
                                "0", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .to_dict()
                        ),
                        "iter": 0,
                    },
                ),
            ],
            [("_phase", 0), ("_phase_count", 2)],
            0.25,
            id="ensemble_smoother_25",
        ),
        pytest.param(
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={
                        **(build_snapshot(["0", "1"]).to_dict()),
                        "iter": 0,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={
                        **(
                            build_partial(["0", "1"])
                            .update_step(
                                "0", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .update_step(
                                "1", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .to_dict()
                        ),
                        "iter": 0,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={
                        **(build_snapshot(["0", "1"]).to_dict()),
                        "iter": 1,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={
                        **(
                            build_partial(["0", "1"])
                            .update_step(
                                "0", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .to_dict()
                        ),
                        "iter": 1,
                    },
                ),
            ],
            [("_phase", 1), ("_phase_count", 2)],
            0.75,
            id="ensemble_smoother_75",
        ),
        pytest.param(
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={
                        **(build_snapshot(["0", "1"]).to_dict()),
                        "iter": 0,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={
                        **(
                            build_partial(["0", "1"])
                            .update_step(
                                "0", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .update_step(
                                "1", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .to_dict()
                        ),
                        "iter": 0,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={
                        **(build_snapshot(["0", "1"]).to_dict()),
                        "iter": 1,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={
                        **(
                            build_partial(["0", "1"])
                            .update_step(
                                "0", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .update_step(
                                "1", "0", Step(status=state.STEP_STATE_SUCCESS)
                            )
                            .to_dict()
                        ),
                        "iter": 1,
                    },
                ),
            ],
            [("_phase", 1), ("_phase_count", 2)],
            1.0,
            id="ensemble_smoother_100",
        ),
    ],
)
def test_tracking_progress(
    monitor_events: List[CloudEvent],
    brm_mutations: List[Tuple[str, Any]],
    expected_progress: float,
    make_mock_ee_monitor,
):
    """Tests progress by providing a list of CloudEvent and a list of
    arguments to apply to setattr(brm) where brm is an actual BaseRunModel
    instance.

    The CloudEvent are provided to the tracker via mocking an Ensemble
    Evaluator Monitor.

    All events leading up to the final update event and end event, are
    discarded. The test asserts that the state of the world is correct only for
    the final update event."""
    brm = BaseRunModel(None)
    with patch("ert_shared.status.tracker.evaluator.create_ee_monitor") as mock_ee:
        mock_ee.return_value.__enter__.return_value = make_mock_ee_monitor(
            monitor_events.copy()
        )
        tracker = EvaluatorTracker(
            brm, "host", "port", 0, 0, next_ensemble_evaluator_wait_time=0.1
        )
        for attr, val in brm_mutations:
            setattr(brm, attr, val)
        tracker_gen = tracker.track()
        update_event = None
        for _ in range(len(monitor_events)):
            update_event = next(tracker_gen)
        assert isinstance(update_event, SnapshotUpdateEvent)
        assert update_event.progress == expected_progress
        brm._phase = brm._phase_count
        assert isinstance(next(tracker_gen), EndEvent)
