import copy
from datetime import datetime as dt

import pytest
from PyQt6.QtCore import QModelIndex
from PyQt6.QtGui import QColor
from pytestqt.qt_compat import qt_api

from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
    FMStepSnapshot,
    RealizationSnapshot,
)
from ert.ensemble_evaluator.state import (
    COLOR_FAILED,
    FORWARD_MODEL_STATE_START,
    REALIZATION_STATE_RUNNING,
)
from ert.gui.model.snapshot import FMStepColorHint, SnapshotModel

from .gui_models_utils import finish_snapshot


@pytest.fixture
def small_snapshot() -> EnsembleSnapshot:
    real = RealizationSnapshot(
        status=REALIZATION_STATE_RUNNING,
        active=True,
        exec_hosts="COMP-01",
        fm_steps={
            "0": FMStepSnapshot(
                start_time=dt.now(),
                end_time=dt.now(),
                name="poly_eval",
                index="0",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage=123,
                max_memory_usage=312,
            ),
            "1": FMStepSnapshot(
                start_time=dt.now(),
                end_time=dt.now(),
                name="poly_postval",
                index="1",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage=123,
                max_memory_usage=312,
            ),
        },
    )
    snapshot = EnsembleSnapshot()
    for i in range(10):
        snapshot.add_realization(str(i), copy.deepcopy(real))

    return snapshot


@pytest.mark.integration_test
@pytest.mark.skip_mac_ci  # slow
def test_using_qt_model_tester(small_snapshot):
    model = SnapshotModel()

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Fatal
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    model._add_snapshot(SnapshotModel.prerender(small_snapshot), "0")
    model._add_snapshot(SnapshotModel.prerender(small_snapshot), "1")
    snapshot = finish_snapshot(SnapshotModel.prerender(small_snapshot))
    model._update_snapshot(SnapshotModel.prerender(snapshot), "0")
    model._update_snapshot(SnapshotModel.prerender(snapshot), "1")


def test_realization_sort_order(full_snapshot):
    model = SnapshotModel()

    model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")

    for i in range(100):
        iter_index = model.index(i, 0, model.index(0, 0, QModelIndex()))

        assert str(i) == str(iter_index.internalPointer().id_)


def test_realization_state_is_queue_finalized_state(fail_snapshot):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(fail_snapshot), "0")
    first_real = model.index(0, 0, model.index(0, 0))

    color, done_count, full_count = model.data(first_real, FMStepColorHint)
    assert color == QColor(*COLOR_FAILED)
    assert done_count == 1
    assert full_count == 1


def test_snapshot_model_data_intact_on_full_update(full_snapshot, fail_snapshot):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")

    first_real = model.index(0, 0, model.index(0, 0))
    assert first_real.internalPointer().children["0"].data["status"] == "Running"
    # Update with a different snapshot, -- data should change accordingly
    model._add_snapshot(SnapshotModel.prerender(fail_snapshot), "0")
    first_real = model.index(0, 0, model.index(0, 0))

    assert first_real.internalPointer().children["0"].data["status"] == "Finished"


@pytest.mark.parametrize(
    "has_exec_hosts, expected_value",
    [
        pytest.param(
            True,
            "COMP-01",
            id="Host assigned",
        ),
        pytest.param(
            False,
            None,
            id="No host assigned",
        ),
    ],
)
def test_snapshot_model_exec_hosts_propagated(
    full_snapshot, fail_snapshot, has_exec_hosts, expected_value
):
    model = SnapshotModel()
    a_snapshot = full_snapshot if has_exec_hosts else fail_snapshot

    model._add_snapshot(SnapshotModel.prerender(a_snapshot), "0")
    model._update_snapshot(SnapshotModel.prerender(a_snapshot), "0")

    first_real = model.index(0, 0, model.index(0, 0))
    assert first_real.internalPointer().data.exec_hosts == expected_value
