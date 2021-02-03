from PyQt5.QtCore import QModelIndex
from ert_gui.model.real_list import RealListModel
from tests.gui.models.conftest import partial_snapshot
from ert_gui.model.snapshot import SnapshotModel
from pytestqt.qt_compat import qt_api


def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    partial = partial_snapshot(full_snapshot)
    source_model = SnapshotModel()

    model = RealListModel(None, 0)
    model.setSourceModel(source_model)

    reporting_mode = (
        qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    )
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(full_snapshot, 0)
    source_model._add_snapshot(full_snapshot, 1)

    source_model._add_partial_snapshot(partial, 0)
    source_model._add_partial_snapshot(partial, 1)

    qtmodeltester.check(model, force_py=True)
