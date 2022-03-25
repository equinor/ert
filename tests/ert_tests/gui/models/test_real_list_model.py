from gui_models_utils import partial_snapshot
from PyQt5.QtCore import QModelIndex
from pytestqt.qt_compat import qt_api

from ert_gui.model.real_list import RealListModel
from ert_gui.model.snapshot import NodeRole, SnapshotModel
from ert.ensemble_evaluator.snapshot import Realization
from ert.ensemble_evaluator.state import (
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_UNKNOWN,
)


def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    source_model = SnapshotModel()

    model = RealListModel(None, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)
    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 1)

    partial = partial_snapshot(full_snapshot)
    source_model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)
    source_model._add_partial_snapshot(SnapshotModel.prerender(partial), 1)

    qtmodeltester.check(model, force_py=True)


def test_change_iter(full_snapshot):
    source_model = SnapshotModel()

    model = RealListModel(None, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)

    assert (
        model.index(0, 0, QModelIndex()).data(NodeRole).data["status"]
        == REALIZATION_STATE_UNKNOWN
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 1)

    model.setIter(1)

    partial = partial_snapshot(full_snapshot)
    partial.update_real("0", Realization(status=REALIZATION_STATE_FINISHED))
    source_model._add_partial_snapshot(SnapshotModel.prerender(partial), 1)

    assert (
        model.index(0, 0, QModelIndex()).data(NodeRole).data["status"]
        == REALIZATION_STATE_FINISHED
    )
