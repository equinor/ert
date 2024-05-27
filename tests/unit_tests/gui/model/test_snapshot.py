import pytest
from PyQt5.QtCore import Qt
from pytestqt.qt_compat import qt_api
from qtpy.QtCore import QModelIndex
from qtpy.QtGui import QColor

from ert.ensemble_evaluator.snapshot import ForwardModel, PartialSnapshot
from ert.ensemble_evaluator.state import (
    COLOR_FAILED,
    COLOR_FINISHED,
    COLOR_PENDING,
    COLOR_RUNNING,
    COLOR_WAITING,
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_RUNNING,
    FORWARD_MODEL_STATE_START,
    REALIZATION_STATE_WAITING,
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


def test_realization_state_matches_display_color(full_snapshot):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(full_snapshot), 0)

    partial = PartialSnapshot(full_snapshot)
    partial.update_forward_model(
        "0", "0", ForwardModel(status=FORWARD_MODEL_STATE_FINISHED)
    )
    partial.update_forward_model(
        "0", "1", ForwardModel(status=FORWARD_MODEL_STATE_FAILURE)
    )
    partial.update_forward_model(
        "0", "2", ForwardModel(status=FORWARD_MODEL_STATE_RUNNING)
    )
    model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)
    first_real = model.index(0, 0, model.index(0, 0))

    queue_color = model.data(first_real, RealStatusColorHint)
    assert queue_color.name() == QColor(*COLOR_RUNNING).name()

    colors = model.data(first_real, RealJobColorHint)
    color_list = [
        QColor(*COLOR_FINISHED).name(),
        QColor(*COLOR_FAILED).name(),
        QColor(*COLOR_RUNNING).name(),
        QColor(*COLOR_PENDING).name(),
    ]
    status_list = [
        FORWARD_MODEL_STATE_FINISHED,
        FORWARD_MODEL_STATE_FAILURE,
        FORWARD_MODEL_STATE_RUNNING,
        FORWARD_MODEL_STATE_START,
    ]

    for i in range(4):
        assert colors[i].name() == color_list[i]

        # verify forward_model coloring and status texts
        status = model.data(model.index(i, 2, first_real), Qt.DisplayRole)
        color = model.data(model.index(i, 2, first_real), Qt.BackgroundRole)
        assert status == status_list[i]
        assert color.name() == color_list[i]


@pytest.mark.parametrize(
    "forward_model_state, expected_states, expected_colors",
    [
        (
            FORWARD_MODEL_STATE_FINISHED,
            [FORWARD_MODEL_STATE_FINISHED, REALIZATION_STATE_WAITING],
            [QColor(*COLOR_FINISHED).name(), QColor(*COLOR_WAITING).name()],
        ),
        (
            FORWARD_MODEL_STATE_RUNNING,
            [FORWARD_MODEL_STATE_RUNNING, FORWARD_MODEL_STATE_START],
            [QColor(*COLOR_RUNNING).name(), QColor(*COLOR_PENDING).name()],
        ),
    ],
)
def test_display_color_changes_when_realization_state_is_not_running(
    forward_model_state, expected_states, expected_colors, waiting_snapshot
):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(waiting_snapshot), 0)

    partial = PartialSnapshot(waiting_snapshot)
    partial.update_forward_model("0", "0", ForwardModel(status=forward_model_state))
    model._add_partial_snapshot(SnapshotModel.prerender(partial), 0)
    first_real = model.index(0, 0, model.index(0, 0))

    queue_color = model.data(first_real, RealStatusColorHint)
    # the queue has to have state 'WAIT' if fm states is to show `Waiting`
    assert queue_color.name() == QColor(*COLOR_WAITING).name()
    colors = model.data(first_real, RealJobColorHint)

    for i in range(2):
        assert colors[i].name() == expected_colors[i]
        status = model.data(model.index(i, 2, first_real), Qt.DisplayRole)
        color = model.data(model.index(i, 2, first_real), Qt.BackgroundRole)
        assert status == expected_states[i]
        assert color.name() == expected_colors[i]
