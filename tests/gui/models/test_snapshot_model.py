from ert_gui.model.snapshot import RealJobColorHint, SnapshotModel
from ert_shared.ensemble_evaluator.entity.snapshot import Job, PartialSnapshot
from ert_shared.status.entity.state import (
    COLOR_PENDING,
    COLOR_RUNNING,
    JOB_STATE_RUNNING,
)
from pytestqt.qt_compat import qt_api
from qtpy.QtCore import QModelIndex
from qtpy.QtGui import QColor
from tests.gui.conftest import partial_snapshot


def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    model = SnapshotModel()

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa, prevent GC
        model, reporting_mode
    )

    model._add_snapshot(full_snapshot, 0)
    model._add_snapshot(full_snapshot, 1)

    partial = partial_snapshot(full_snapshot)
    model._add_partial_snapshot(partial, 0)
    model._add_partial_snapshot(partial, 1)

    qtmodeltester.check(model, force_py=True)


def test_realization_sort_order(full_snapshot):
    model = SnapshotModel()

    model._add_snapshot(full_snapshot, 0)

    for i in range(0, 100):
        iter_index = model.index(i, 0, model.index(0, 0, QModelIndex()))

        assert str(i) == iter_index.internalPointer().id, print(
            i, iter_index.internalPointer()
        )


def test_realization_job_hint(full_snapshot):
    model = SnapshotModel()
    model._add_snapshot(full_snapshot, 0)

    partial = PartialSnapshot(full_snapshot)
    partial.update_job("0", "0", "0", "0", Job(status=JOB_STATE_RUNNING))
    model._add_partial_snapshot(partial, 0)

    first_real = model.index(0, 0, model.index(0, 0))
    colors = model.data(first_real, RealJobColorHint)
    assert colors[0].name() == QColor(*COLOR_RUNNING).name()
    assert colors[1].name() == QColor(*COLOR_PENDING).name()
