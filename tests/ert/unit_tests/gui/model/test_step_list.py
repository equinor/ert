from datetime import datetime
from unittest.mock import patch

import pytest
from dateutil import tz
from PyQt6.QtCore import QModelIndex
from pytestqt.qt_compat import qt_api

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator.snapshot import FMStepSnapshot
from ert.ensemble_evaluator.state import (
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_RUNNING,
    FORWARD_MODEL_STATE_START,
)
from ert.gui.model.fm_step_list import FMStepListProxyModel
from ert.gui.model.snapshot import DURATION, FM_STEP_COLUMNS, SnapshotModel

from .gui_models_utils import finish_snapshot


def _id_to_col(identifier):
    return FM_STEP_COLUMNS.index(identifier)


def test_using_qt_model_tester(full_snapshot):
    snapshot = finish_snapshot(full_snapshot)
    source_model = SnapshotModel()

    model = FMStepListProxyModel(None, 0, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Fatal
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")
    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "1")

    source_model._update_snapshot(SnapshotModel.prerender(snapshot), "0")
    source_model._update_snapshot(SnapshotModel.prerender(snapshot), "1")


@pytest.mark.requires_window_manager
def test_changes(full_snapshot):
    source_model = SnapshotModel()

    model = FMStepListProxyModel(None, 0, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Fatal
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")
    assert model.index(0, _id_to_col(ids.STATUS)).data() == FORWARD_MODEL_STATE_START

    snapshot = full_snapshot
    start_time = datetime(year=2020, month=10, day=27, hour=12)
    end_time = datetime(year=2020, month=10, day=28, hour=13)
    snapshot.update_fm_step(
        "0",
        "0",
        fm_step=FMStepSnapshot(
            status=FORWARD_MODEL_STATE_FAILURE,
            start_time=start_time,
            end_time=end_time,
        ),
    )
    source_model._update_snapshot(SnapshotModel.prerender(snapshot), "0")
    assert (
        model.index(0, _id_to_col(DURATION), QModelIndex()).data() == "1 day, 1:00:00"
    )
    assert (
        model.index(0, _id_to_col(ids.STATUS), QModelIndex()).data()
        == FORWARD_MODEL_STATE_FAILURE
    )


@pytest.mark.requires_window_manager
@pytest.mark.parametrize("timezone", [(None), tz.gettz("UTC")])
@patch("ert.gui.model.snapshot.datetime", wraps=datetime)
def test_duration(mock_datetime, timezone, full_snapshot):
    source_model = SnapshotModel()

    model = FMStepListProxyModel(None, 0, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Fatal
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")
    assert (
        model.index(0, _id_to_col(ids.STATUS), QModelIndex()).data()
        == FORWARD_MODEL_STATE_START
    )

    snapshot = full_snapshot
    start_time = datetime(year=2020, month=10, day=27, hour=12, tzinfo=timezone)
    # mock only datetime.now()
    mock_datetime.now.return_value = datetime(
        year=2020,
        month=10,
        day=28,
        hour=13,
        minute=12,
        second=11,
        microsecond=5,  # Note that microseconds are intended to be removed
        tzinfo=timezone,
    )
    snapshot.update_fm_step(
        "0",
        "2",
        fm_step=FMStepSnapshot(
            status=FORWARD_MODEL_STATE_RUNNING,
            start_time=start_time,
        ),
    )
    source_model._update_snapshot(SnapshotModel.prerender(snapshot), "0")
    assert (
        model.index(2, _id_to_col(DURATION), QModelIndex()).data() == "1 day, 1:12:11"
    )
    mock_datetime.now.assert_called_once_with(timezone)


@pytest.mark.requires_window_manager
def test_no_cross_talk(full_snapshot):
    source_model = SnapshotModel()

    model = FMStepListProxyModel(None, 0, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Fatal
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")
    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "1")

    # Test that changes to iter=1 does not bleed into iter=0
    snapshot = full_snapshot
    snapshot.update_fm_step(
        "0", "0", fm_step=FMStepSnapshot(status=FORWARD_MODEL_STATE_FAILURE)
    )
    source_model._update_snapshot(SnapshotModel.prerender(snapshot), "1")
    assert (
        model.index(0, _id_to_col(ids.STATUS), QModelIndex()).data()
        == FORWARD_MODEL_STATE_START
    )

    model.set_real(1, 0)
    assert (
        model.index(0, _id_to_col(ids.STATUS), QModelIndex()).data()
        == FORWARD_MODEL_STATE_FAILURE
    )
