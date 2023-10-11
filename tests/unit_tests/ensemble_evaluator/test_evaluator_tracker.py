import math
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from cloudevents.http.event import CloudEvent

import ert.ensemble_evaluator.identifiers as ids
from ert.ensemble_evaluator import EvaluatorTracker, state
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.event import EndEvent, SnapshotUpdateEvent
from ert.ensemble_evaluator.snapshot import PartialSnapshot, SnapshotBuilder, Step
from ert.run_models import BaseRunModel


def build_snapshot(real_list: Optional[List[str]] = None):
    if real_list is None:
        # passing ["0"] is required
        real_list = ["0"]
    return (
        SnapshotBuilder()
        .add_step(status=state.STEP_STATE_UNKNOWN)
        .build(real_list, state.REALIZATION_STATE_UNKNOWN)
    )


def build_partial(real_list: Optional[List[str]] = None):
    if real_list is None:
        real_list = ["0"]
    return PartialSnapshot(build_snapshot(real_list))


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
    "run_model, monitor_events,brm_mutations,expected_progress",
    [
        pytest.param(
            BaseRunModel,
            [
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={
                        **(build_snapshot(["0", "1"]).to_dict()),
                        "iter": 2,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={
                        **(
                            build_partial(["0", "1"])
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .to_dict()
                        ),
                        "iter": 2,
                    },
                ),
            ],
            [("_phase_count", 1)],
            [0, 0.5],
            id="ensemble_experiment_50",
        ),
        pytest.param(
            BaseRunModel,
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
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .to_dict()
                        ),
                        "iter": 0,
                    },
                ),
            ],
            [("_phase_count", 1)],
            [0, 0.5],
            id="ensemble_experiment_50",
        ),
        pytest.param(
            BaseRunModel,
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
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .to_dict()
                        ),
                        "iter": 0,
                    },
                ),
            ],
            [("_phase_count", 2)],
            [0, 0.25],
            id="ensemble_smoother_25",
        ),
        pytest.param(
            BaseRunModel,
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
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .update_step("1", Step(status=state.STEP_STATE_SUCCESS))
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
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .to_dict()
                        ),
                        "iter": 1,
                    },
                ),
            ],
            [("_phase_count", 2)],
            [
                0,
                0.5,
                0.5,
                0.75,
            ],
            id="ensemble_smoother_75",
        ),
        pytest.param(
            BaseRunModel,
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
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .update_step("1", Step(status=state.STEP_STATE_SUCCESS))
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
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .update_step("1", Step(status=state.STEP_STATE_SUCCESS))
                            .to_dict()
                        ),
                        "iter": 1,
                    },
                ),
            ],
            [("_phase_count", 2)],
            [
                0,
                0.5,
                0.5,
                1.0,
            ],
            id="ensemble_smoother_100",
        ),
        pytest.param(
            BaseRunModel,
            [
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
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .update_step("1", Step(status=state.STEP_STATE_SUCCESS))
                            .to_dict()
                        ),
                        "iter": 1,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT},
                    data={
                        **(build_snapshot(["0", "1"]).to_dict()),
                        "iter": 2,
                    },
                ),
                CloudEvent(
                    {"source": "/", "type": ids.EVTYPE_EE_SNAPSHOT_UPDATE},
                    data={
                        **(
                            build_partial(["0", "1"])
                            .update_step("0", Step(status=state.STEP_STATE_SUCCESS))
                            .update_step("1", Step(status=state.STEP_STATE_SUCCESS))
                            .to_dict()
                        ),
                        "iter": 2,
                    },
                ),
            ],
            [("_phase_count", 3)],
            [
                0.3333,
                0.6666,
                0.6666,
                1.0,
            ],
            id="ensemble_smoother_100",
        ),
    ],
)
def test_tracking_progress(
    run_model: BaseRunModel,
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

    PartialSnapshots allow realizations to progress, while iterating "iter" in
    CloudEvents allows phases to progress. Such progress should happen
    when events are yielded by the tracker. This combined progress is tested.

    The final update event and end event is also tested."""
    arg_mock = MagicMock()
    arg_mock.random_seed = None
    run_model.validate = MagicMock()
    brm = run_model(arg_mock, None, None, None, None, None)
    ee_config = EvaluatorServerConfig(
        custom_port_range=range(1024, 65535),
        custom_host="127.0.0.1",
        use_token=False,
        generate_cert=False,
    )
    with patch("ert.ensemble_evaluator.evaluator_tracker.Monitor") as mock_ee:
        mock_ee.return_value.__enter__.return_value = make_mock_ee_monitor(
            monitor_events.copy()
        )
        tracker = EvaluatorTracker(
            brm, ee_config.get_connection_info(), next_ensemble_evaluator_wait_time=0.1
        )
        for attr, val in brm_mutations:
            setattr(brm, attr, val)
        tracker_gen = tracker.track()
        update_event = None
        for i in range(len(monitor_events)):
            update_event = next(tracker_gen)
            assert math.isclose(
                update_event.progress, expected_progress[i], rel_tol=0.0001
            )
        assert isinstance(update_event, SnapshotUpdateEvent)
        brm._phase = brm._phase_count
        assert isinstance(next(tracker_gen), EndEvent)
