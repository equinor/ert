from ert_shared.status.entity.state import (
    JOB_STATE_FINISHED,
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_UNKNOWN,
)
from ert_shared.ensemble_evaluator.entity.snapshot import PartialSnapshot, Realization
from PyQt5.QtCore import QModelIndex
from ert_gui.model.progress_proxy import ProgressProxyModel
from tests.gui.conftest import partial_snapshot
from ert_gui.model.snapshot import ProgressRole, SnapshotModel
from pytestqt.qt_compat import qt_api


def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    partial = partial_snapshot(full_snapshot)
    source_model = SnapshotModel()

    model = ProgressProxyModel(source_model, parent=None)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(full_snapshot, 0)
    source_model._add_snapshot(full_snapshot, 1)

    source_model._add_partial_snapshot(partial, 0)
    source_model._add_partial_snapshot(partial, 1)

    qtmodeltester.check(model, force_py=True)


def test_progression(full_snapshot):
    source_model = SnapshotModel()
    model = ProgressProxyModel(source_model, parent=None)

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    source_model._add_snapshot(full_snapshot, 0)

    assert model.data(model.index(0, 0, QModelIndex()), ProgressRole) == {
        "nr_reals": 100,
        "status": {REALIZATION_STATE_UNKNOWN: 100},
    }

    partial = PartialSnapshot(full_snapshot)
    partial.update_real("0", Realization(status=REALIZATION_STATE_FINISHED))
    source_model._add_partial_snapshot(partial, 0)

    assert model.data(model.index(0, 0, QModelIndex()), ProgressRole) == {
        "nr_reals": 100,
        "status": {REALIZATION_STATE_UNKNOWN: 99, REALIZATION_STATE_FINISHED: 1},
    }
