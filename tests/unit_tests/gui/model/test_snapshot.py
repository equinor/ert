from pytestqt.qt_compat import qt_api
from qtpy.QtCore import QModelIndex
from qtpy.QtGui import QColor

from ert.ensemble_evaluator.snapshot import ForwardModel, PartialSnapshot
from ert.ensemble_evaluator.state import (
    COLOR_PENDING,
    COLOR_RUNNING,
    FORWARD_MODEL_STATE_RUNNING,
)
from ert.gui.model.snapshot import RealJobColorHint, SnapshotModel

from .gui_models_utils import partial_snapshot


def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    model = SnapshotModel()

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)
    model._add_snapshot(SnapshotModel.prerender(full_snapshot), 1)

    partial = partial_snapshot(SnapshotModel.prerender(full_snapshot))
    model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)
    model._add_partial_snapshot(SnapshotModel.prerender(partial), 1)

    qtmodeltester.check(model, force_py=True)


def test_realization_sort_order(full_snapshot):
    model = SnapshotModel()

    model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)

    for i in range(0, 100):
        iter_index = model.index(i, 0, model.index(0, 0, QModelIndex()))

        assert str(i) == iter_index.internalPointer().id, print(
            i, iter_index.internalPointer()
        )


def test_realization_job_hint(full_snapshot):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)

    partial = PartialSnapshot(full_snapshot)
    partial.update_job("0", "0", ForwardModel(status=FORWARD_MODEL_STATE_RUNNING))
    model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)

    first_real = model.index(0, 0, model.index(0, 0))
    colors = model.data(first_real, RealJobColorHint)
    assert colors[0].name() == QColor(*COLOR_RUNNING).name()
    assert colors[1].name() == QColor(*COLOR_PENDING).name()
