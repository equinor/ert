import pytest
from pytestqt.qt_compat import qt_api
from qtpy.QtCore import QModelIndex
from qtpy.QtGui import QColor

from ert.ensemble_evaluator.snapshot import ForwardModel, PartialSnapshot
from ert.ensemble_evaluator.state import (
    COLOR_FAILED,
    COLOR_PENDING,
    COLOR_RUNNING,
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_RUNNING,
    FORWARD_MODEL_STATE_START,
)
from ert.gui.model.snapshot import RealJobColorHint, RealStatusColorHint, SnapshotModel

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

        assert str(i) == str(iter_index.internalPointer().id_), print(
            i, iter_index.internalPointer()
        )


def test_realization_state_is_overridden_by_queue_finalized_state(fail_snapshot):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(fail_snapshot), 0)

    partial = PartialSnapshot(fail_snapshot)
    partial.update_forward_model(
        "0", "0", ForwardModel(status=FORWARD_MODEL_STATE_FINISHED)
    )

    model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)
    first_real = model.index(0, 0, model.index(0, 0))

    queue_color = model.data(first_real, RealStatusColorHint)
    assert queue_color == QColor(*COLOR_FAILED)
    color, done_count, full_count = model.data(first_real, RealJobColorHint)
    assert color == QColor(*COLOR_FAILED)
    assert done_count == 1
    assert full_count == 1


@pytest.mark.parametrize(
    "first_state, second_state, expected_color",
    [
        (
            FORWARD_MODEL_STATE_FINISHED,
            FORWARD_MODEL_STATE_START,
            QColor(*COLOR_PENDING),
        ),
        (
            FORWARD_MODEL_STATE_FINISHED,
            FORWARD_MODEL_STATE_RUNNING,
            QColor(*COLOR_RUNNING),
        ),
        (
            FORWARD_MODEL_STATE_FINISHED,
            FORWARD_MODEL_STATE_FAILURE,
            QColor(*COLOR_FAILED),
        ),
        (
            FORWARD_MODEL_STATE_RUNNING,
            FORWARD_MODEL_STATE_START,
            QColor(*COLOR_RUNNING),
        ),
        (
            FORWARD_MODEL_STATE_FAILURE,
            FORWARD_MODEL_STATE_START,
            QColor(*COLOR_FAILED),
        ),
    ],
)
def test_display_color_changes_to_more_important_state(
    first_state, second_state, expected_color, full_snapshot
):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)

    partial = PartialSnapshot(full_snapshot)
    partial.update_forward_model("0", "0", ForwardModel(status=first_state))
    partial.update_forward_model("0", "1", ForwardModel(status=second_state))
    model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)
    first_real = model.index(0, 0, model.index(0, 0))

    color, _, _ = model.data(first_real, RealJobColorHint)
    assert color == expected_color
