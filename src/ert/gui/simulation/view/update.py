from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ert.run_models import (
    RunModelEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
)


class UpdateWidget(QWidget):
    def __init__(self, iteration: int, parent=None) -> None:
        super().__init__(parent)

        self._iteration = iteration

        self._progress_label = QLabel("Progress:")
        self._progress_msg = QLabel()

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(0)

        self._msg_list = QListWidget()
        self._msg_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        _layout = QVBoxLayout()
        _layout.setContentsMargins(100, 20, 100, 20)

        _top_layout = QHBoxLayout()
        _top_layout.addWidget(self._progress_label)
        _top_layout.addWidget(self._progress_msg)
        _top_layout.addStretch()

        _layout.addLayout(_top_layout)
        _layout.addWidget(self._progress_bar)
        _layout.addWidget(self._msg_list, alignment=Qt.AlignTop)
        _layout.addStretch()

        self.setLayout(_layout)

    @property
    def iteration(self) -> int:
        return self._iteration

    @Slot()
    def begin(self) -> None:
        pass

    @Slot()
    def end(self) -> None:
        self._progress_msg.setText("Update completed")
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(1)
        self._progress_bar.setValue(1)

    @Slot(RunModelEvent)
    def update_status(self, event: RunModelEvent) -> None:
        if isinstance(event, RunModelStatusEvent):
            item = QListWidgetItem()
            item.setText(event.msg)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._msg_list.addItem(item)
        elif isinstance(event, RunModelTimeEvent):
            self._progress_msg.setText(
                f"Estimated remaining time for current step {event.remaining_time:.2f}s"
            )
