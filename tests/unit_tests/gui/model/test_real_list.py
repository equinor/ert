from PyQt5.QtCore import QModelIndex
from pytestqt.qt_compat import qt_api

from ert.ensemble_evaluator.state import (
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_UNKNOWN,
)
from ert.gui.model.real_list import RealListModel
from ert.gui.model.snapshot import NodeRole, SnapshotModel

from .gui_models_utils import partial_snapshot


def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    source_model = SnapshotModel()

    model = RealListModel(None, 0)
    model.setSourceModel(source_model)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    # pylint: disable=unused-variable
    tester = qt_api.QtTest.QAbstractItemModelTester(model, reporting_mode)  # noqa: F841

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
    # pylint: disable=unused-variable
    tester = qt_api.QtTest.QAbstractItemModelTester(model, reporting_mode)  # noqa: F841

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)

    assert (
        model.index(0, 0, QModelIndex()).data(NodeRole).data["status"]
        == REALIZATION_STATE_UNKNOWN
    )

    source_model._add_snapshot(SnapshotModel.prerender(full_snapshot), 1)

    model.setIter(1)

    partial = partial_snapshot(full_snapshot)
    partial._realization_states["0"].update({"status": REALIZATION_STATE_FINISHED})
    source_model._add_partial_snapshot(SnapshotModel.prerender(partial), 1)

    assert (
        model.index(0, 0, QModelIndex()).data(NodeRole).data["status"]
        == REALIZATION_STATE_FINISHED
    )
