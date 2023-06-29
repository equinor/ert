import pytest
from PyQt5.QtCore import QModelIndex
from pytestqt.qt_compat import qt_api

from ert.ensemble_evaluator.snapshot import PartialSnapshot
from ert.ensemble_evaluator.state import (
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_UNKNOWN,
)
from ert.gui.model.progress_proxy import ProgressProxyModel
from ert.gui.model.snapshot import ProgressRole, SnapshotModel

from .gui_models_utils import partial_snapshot


def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    source_model = SnapshotModel()

    model = ProgressProxyModel(source_model, parent=None)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    # pylint: disable=unused-variable
    tester = qt_api.QtTest.QAbstractItemModelTester(model, reporting_mode)  # noqa: F841

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)
    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 1)

    partial = partial_snapshot(full_snapshot)
    source_model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)
    source_model._add_partial_snapshot(SnapshotModel.prerender(partial), 1)

    qtmodeltester.check(model, force_py=True)


@pytest.mark.requires_window_manager
def test_progression(full_snapshot):
    source_model = SnapshotModel()
    model = ProgressProxyModel(source_model, parent=None)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    # pylint: disable=unused-variable
    tester = qt_api.QtTest.QAbstractItemModelTester(model, reporting_mode)  # noqa: F841

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)

    assert model.data(model.index(0, 0, QModelIndex()), ProgressRole) == {
        "nr_reals": 100,
        "status": {REALIZATION_STATE_UNKNOWN: 100},
    }

    partial = PartialSnapshot(full_snapshot)
    partial._realization_states["0"].update({"status": REALIZATION_STATE_FINISHED})
    source_model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)

    assert model.data(model.index(0, 0, QModelIndex()), ProgressRole) == {
        "nr_reals": 100,
        "status": {REALIZATION_STATE_UNKNOWN: 99, REALIZATION_STATE_FINISHED: 1},
    }


@pytest.mark.requires_window_manager
def test_progression_start_iter_not_zero(full_snapshot):
    source_model = SnapshotModel()
    model = ProgressProxyModel(source_model, parent=None)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    # pylint: disable=unused-variable
    tester = qt_api.QtTest.QAbstractItemModelTester(model, reporting_mode)  # noqa: F841

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 1)

    assert model.data(model.index(0, 0, QModelIndex()), ProgressRole) == {
        "nr_reals": 100,
        "status": {REALIZATION_STATE_UNKNOWN: 100},
    }

    partial = PartialSnapshot(full_snapshot)
    partial._realization_states["0"].update({"status": REALIZATION_STATE_FINISHED})
    source_model._add_partial_snapshot(SnapshotModel.prerender(partial), 1)

    assert model.data(model.index(0, 0, QModelIndex()), ProgressRole) == {
        "nr_reals": 100,
        "status": {REALIZATION_STATE_UNKNOWN: 99, REALIZATION_STATE_FINISHED: 1},
    }
