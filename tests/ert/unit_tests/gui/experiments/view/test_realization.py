from datetime import datetime as dt

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, QModelIndex, QSize, Qt
from PyQt6.QtGui import QHelpEvent
from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem

from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
)
from ert.ensemble_evaluator.state import (
    FORWARD_MODEL_STATE_START,
    REALIZATION_STATE_UNKNOWN,
)
from ert.gui.experiments.view.realization import RealizationWidget
from ert.gui.model.node import ForwardModelStepNode, IterNode, RealNode, RootNode
from ert.gui.model.snapshot import SnapshotModel
from tests.ert import SnapshotBuilder


@pytest.fixture
def small_snapshot() -> EnsembleSnapshot:
    builder = SnapshotBuilder()
    for i in range(2):
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
    real_ids = [str(i) for i in range(5)]
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


@pytest.mark.integration_test
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


@pytest.mark.integration_test
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
        return isinstance(
            node, ForwardModelStepNode | RealNode | IterNode | RootNode
        ) and str(node.id_) == str(selection_id)

    with qtbot.waitSignal(
        widget.itemClicked, timeout=30000, check_params_cb=check_selection_cb
    ):
        qtbot.mouseClick(
            widget._real_view.viewport(),
            Qt.MouseButton.LeftButton,
            pos=selection_rect.center(),
        )


@pytest.mark.integration_test
def test_realization_hover_yields_tooltip(full_snapshot, qtbot):
    it = 0
    widget = RealizationWidget(it)

    qtbot.addWidget(widget)

    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(full_snapshot), str(it))
    widget.setSnapshotModel(model)
    model._update_snapshot(full_snapshot, str(it))

    widget.resize(800, 600)
    widget.move(0, 0)

    with qtbot.waitActive(widget, timeout=5000):
        widget.show()

    realization_selection_id = 22  # randomly selected realization
    selection_rect = widget._real_view.rectForIndex(
        widget._real_list_model.index(realization_selection_id, 0, QModelIndex())
    )

    help_event = QHelpEvent(
        QEvent.Type.ToolTip,
        selection_rect.center(),
        widget.mapToGlobal(selection_rect.center()),
    )

    with qtbot.waitSignal(widget.triggeredTooltipTextDisplay, timeout=2000) as tooltip:
        QCoreApplication.postEvent(widget._real_view.viewport(), help_event)

    assert tooltip.signal_triggered
    assert "Maximum memory" in tooltip.args[0]
