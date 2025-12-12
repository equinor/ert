from PyQt6.QtCore import QModelIndex
from pytestqt.qt_compat import qt_api

from ert.ensemble_evaluator.state import (
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_RUNNING,
)
from ert.gui.model.real_list import RealListModel
from ert.gui.model.snapshot import NodeRole, SnapshotModel

from .gui_models_utils import finish_snapshot


def test_using_qt_model_tester(full_snapshot):
    source_model = SnapshotModel()

    model = RealListModel(None, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Fatal
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")
    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "1")

    snapshot = finish_snapshot(full_snapshot)
    source_model._update_snapshot(SnapshotModel.prerender(snapshot), "0")
    source_model._update_snapshot(SnapshotModel.prerender(snapshot), "1")


def test_change_iter(full_snapshot):
    source_model = SnapshotModel()

    model = RealListModel(None, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Fatal
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")

    assert (
        model.index(0, 0, QModelIndex()).data(NodeRole).data.status
        == REALIZATION_STATE_RUNNING
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), "1")

    model.setIter(1)

    snapshot = finish_snapshot(snapshot=full_snapshot)
    snapshot._realization_snapshots["0"].update({"status": REALIZATION_STATE_FINISHED})
    source_model._update_snapshot(SnapshotModel.prerender(snapshot), "1")

    assert (
        model.index(0, 0, QModelIndex()).data(NodeRole).data.status
        == REALIZATION_STATE_FINISHED
    )
