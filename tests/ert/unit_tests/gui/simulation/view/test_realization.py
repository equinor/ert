from datetime import datetime as dt

import pytest
from qtpy import QtCore
from qtpy.QtCore import QModelIndex, QSize
from qtpy.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
)
from ert.ensemble_evaluator.state import (
    FORWARD_MODEL_STATE_START,
    REALIZATION_STATE_UNKNOWN,
)
from ert.gui.model.node import _Node
from ert.gui.model.snapshot import SnapshotModel
from ert.gui.simulation.view.realization import RealizationWidget
from tests.ert import SnapshotBuilder


@pytest.fixture
def small_snapshot() -> EnsembleSnapshot:
    builder = SnapshotBuilder()
    for i in range(0, 2):
        builder.add_fm_step(
            fm_step_id=str(i),
            index=str(i),
            name=f"job_{i}",
            current_memory_usage="500",
            max_memory_usage="1000",
            status=FORWARD_MODEL_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=dt(1999, 1, 1),
            end_time=dt(2019, 1, 1),
        )
    real_ids = [str(i) for i in range(0, 5)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)


class MockDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._size = QSize(50, 50)
        self._max_id = 0

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        self._max_id = max(int(index.internalPointer().id_), self._max_id)

    def sizeHint(self, option, index) -> QSize:
        return self._size


def test_delegate_drawing_count(small_snapshot, qtbot):
    it = 0
    widget = RealizationWidget(it)

    qtbot.addWidget(widget)

    with qtbot.waitActive(widget, timeout=30000):
        model = SnapshotModel()
        model._add_snapshot(SnapshotModel.prerender(small_snapshot), str(it))

        widget.setSnapshotModel(model)

        widget.move(0, 0)
        widget.resize(640, 480)

        # mock delegate for counting how many times we draw delegates
        mock_delegate = MockDelegate()
        widget._real_view.setItemDelegate(mock_delegate)

        widget.show()
        qtbot.wait(1000)
        print(mock_delegate._max_id)
        qtbot.waitUntil(
            lambda: mock_delegate._max_id == len(small_snapshot.reals) - 1,
            timeout=30000,
        )


def test_selection_success(large_snapshot, qtbot):
    it = 0
    widget = RealizationWidget(it)

    qtbot.addWidget(widget)

    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(large_snapshot), str(it))

    widget.setSnapshotModel(model)

    widget.resize(800, 600)
    widget.move(0, 0)

    with qtbot.waitActive(widget, timeout=30000):
        widget.show()

    selection_id = 22
    selection_rect = widget._real_view.rectForIndex(
        widget._real_list_model.index(selection_id, 0, QModelIndex())
    )

    def check_selection_cb(index):
        node = index.internalPointer()
        return isinstance(node, _Node) and str(node.id_) == str(selection_id)

    with qtbot.waitSignal(
        widget.itemClicked, timeout=30000, check_params_cb=check_selection_cb
    ):
        qtbot.mouseClick(
            widget._real_view.viewport(),
            QtCore.Qt.LeftButton,
            pos=selection_rect.center(),
        )
