import uuid
from collections import defaultdict
from datetime import UTC
from datetime import datetime as dt

import pytest

from ert.analysis.event import DataSection
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.snapshot import EnsembleSnapshotMetadata
from ert.run_models.event import (
    AnalysisStatusEvent,
    AnalysisTimeEvent,
    CorruptStatusSnapshotError,
    EndEvent,
    FullSnapshotEvent,
    RunModelDataEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
    SnapshotUpdateEvent,
    load_status_snapshot_event,
    status_event_from_json,
    status_event_to_json,
)
from tests.ert.utils import SnapshotBuilder

METADATA = EnsembleSnapshotMetadata(
    fm_step_status=defaultdict(dict),
    real_status={},
    sorted_real_ids=[],
    sorted_fm_step_ids=defaultdict(list),
)


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            FullSnapshotEvent(
                snapshot=(
                    SnapshotBuilder(metadata=METADATA)
                    .add_fm_step(
                        fm_step_id="0",
                        index="0",
                        name="fm_step_0",
                        status=state.FORWARD_MODEL_STATE_START,
                        current_memory_usage="500",
                        max_memory_usage="1000",
                        stdout="job_fm_step_0.stdout",
                        stderr="job_fm_step_0.stderr",
                        start_time=dt(1999, 1, 1, tzinfo=UTC),
                    )
                    .add_fm_step(
                        fm_step_id="1",
                        index="1",
                        name="fm_step_1",
                        status=state.FORWARD_MODEL_STATE_START,
                        current_memory_usage="500",
                        max_memory_usage="1000",
                        stdout="job_fm_step_1.stdout",
                        stderr="job_fm_step_1.stderr",
                        start_time=dt(1999, 1, 1, tzinfo=UTC),
                        end_time=None,
                    )
                    .build(
                        real_ids=["0", "1"],
                        status=state.REALIZATION_STATE_UNKNOWN,
                        start_time=dt(1999, 1, 1, tzinfo=UTC),
                        exec_hosts="12121.121",
                        message="Some message",
                    )
                ),
                iteration_label="Foo",
                total_iterations=1,
                progress=0.25,
                realization_count=4,
                status_count={"Finished": 1, "Pending": 2, "Unknown": 1},
                iteration=0,
            ),
            id="FullSnapshotEvent",
        ),
        pytest.param(
            SnapshotUpdateEvent(
                snapshot=SnapshotBuilder(metadata=METADATA)
                .add_fm_step(
                    fm_step_id="0",
                    index="0",
                    status=state.FORWARD_MODEL_STATE_FINISHED,
                    name="fm_step_0",
                    end_time=dt(2019, 1, 1, tzinfo=UTC),
                )
                .build(
                    real_ids=["1"],
                    status=state.REALIZATION_STATE_RUNNING,
                ),
                iteration_label="Foo",
                total_iterations=1,
                progress=0.5,
                realization_count=4,
                status_count={"Finished": 2, "Running": 1, "Unknown": 1},
                iteration=0,
            ),
            id="SnapshotUpdateEvent1",
        ),
        pytest.param(
            SnapshotUpdateEvent(
                snapshot=SnapshotBuilder(metadata=METADATA)
                .add_fm_step(
                    fm_step_id="1",
                    index="1",
                    status=state.FORWARD_MODEL_STATE_FAILURE,
                    name="fm_step_1",
                )
                .build(
                    real_ids=["0"],
                    status=state.REALIZATION_STATE_FAILED,
                    end_time=dt(2019, 1, 1, tzinfo=UTC),
                ),
                iteration_label="Foo",
                total_iterations=1,
                progress=0.5,
                realization_count=4,
                status_count={"Finished": 2, "Failed": 1, "Unknown": 1},
                iteration=0,
            ),
            id="SnapshotUpdateEvent2",
        ),
        pytest.param(AnalysisStatusEvent(msg="hello"), id="AnalysisStatusEvent"),
        pytest.param(
            AnalysisStatusEvent(msg="hello", detail=True),
            id="AnalysisStatusEvent_detail",
        ),
        pytest.param(
            AnalysisTimeEvent(remaining_time=22.2, elapsed_time=200.42),
            id="AnalysisTimeEvent",
        ),
        pytest.param(EndEvent(failed=False, msg=""), id="EndEvent"),
        pytest.param(
            RunModelStatusEvent(iteration=1, run_id=uuid.uuid1(), msg="Hello"),
            id="RunModelStatusEvent",
        ),
        pytest.param(
            RunModelStatusEvent(
                iteration=1, run_id=uuid.uuid1(), msg="Hello", detail=True
            ),
            id="RunModelStatusEvent_detail",
        ),
        pytest.param(
            RunModelTimeEvent(
                iteration=1,
                run_id=uuid.uuid1(),
                remaining_time=10.42,
                elapsed_time=100.42,
            ),
            id="RunModelTimeEvent",
        ),
        pytest.param(
            RunModelUpdateBeginEvent(iteration=2, run_id=uuid.uuid1()),
            id="RunModelUpdateBeginEvent",
        ),
        pytest.param(
            RunModelDataEvent(
                iteration=1,
                run_id=uuid.uuid1(),
                name="Micky",
                data=DataSection(
                    header=["Some", "string", "elements"],
                    data=[["a", 1.1, "b"], ["c", 3.0]],
                    extra={"a": "b", "c": "d"},
                ),
            ),
            id="RunModelDataEvent",
        ),
        pytest.param(
            RunModelUpdateEndEvent(
                iteration=3,
                run_id=uuid.uuid1(),
                data=DataSection(
                    header=["Some", "string", "elements"],
                    data=[["a", 1.1, "b"], ["c", 3.0]],
                    extra={"a": "b", "c": "d"},
                ),
            ),
            id="RunModelUpdateEndEvent",
        ),
    ],
)
def test_status_event_serialization(event):
    json_res = status_event_to_json(event)
    round_trip_event = status_event_from_json(json_res)
    assert event == round_trip_event


def _build_full_snapshot_event() -> FullSnapshotEvent:
    return FullSnapshotEvent(
        snapshot=SnapshotBuilder(metadata=METADATA)
        .add_fm_step(
            fm_step_id="0",
            index="0",
            name="fm_step_0",
            status=state.FORWARD_MODEL_STATE_FINISHED,
        )
        .build(
            real_ids=["0", "1"],
            status=state.REALIZATION_STATE_FINISHED,
        ),
        iteration_label="Foo",
        total_iterations=1,
        progress=1.0,
        realization_count=2,
        status_count={"Finished": 2},
        iteration=0,
    )


def test_that_persisted_snapshot_is_loaded_back_into_a_full_snapshot_event(tmp_path):
    event = _build_full_snapshot_event()
    path = tmp_path / "snapshot_0.json"
    path.write_bytes(status_event_to_json(event).encode("utf-8"))

    loaded = load_status_snapshot_event(path)

    assert loaded == event


def test_that_loading_a_missing_snapshot_returns_none(tmp_path):
    assert load_status_snapshot_event(tmp_path / "does_not_exist.json") is None


def test_that_loading_an_unparseable_snapshot_raises_corrupt_error(tmp_path):
    path = tmp_path / "snapshot_0.json"
    path.write_text("not valid json", encoding="utf-8")

    with pytest.raises(CorruptStatusSnapshotError, match="Could not parse"):
        load_status_snapshot_event(path)


def test_that_loading_a_snapshot_of_the_wrong_event_type_raises_corrupt_error(tmp_path):
    path = tmp_path / "snapshot_0.json"
    path.write_text(
        status_event_to_json(EndEvent(failed=False, msg="done")), encoding="utf-8"
    )

    with pytest.raises(CorruptStatusSnapshotError, match="FullSnapshotEvent"):
        load_status_snapshot_event(path)
