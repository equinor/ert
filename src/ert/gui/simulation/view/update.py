from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class UpdateWidget(QWidget):
    def __init__(self, iteration: int, parent=None) -> None:
        super().__init__(parent)

        self._iteration = iteration

        self._progress_label = QLabel("Progress")

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(False)

        self._status_text = QLabel()

        _layout = QVBoxLayout()
        _layout.setContentsMargins(100, 20, 100, 20)
        _layout.addWidget(self._progress_label)
        _layout.addWidget(self._progress_bar)
        _layout.addWidget(self._status_text, alignment=Qt.AlignTop)
        _layout.addStretch()

        self.setLayout(_layout)

    @property
    def iteration(self) -> int:
        return self._iteration

    @Slot()
    def begin(self) -> None:
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(0)

    @Slot()
    def end(self) -> None:
        self._progress_label.setText("Progress - Update completed")
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(1)
        self._progress_bar.setValue(1)

    @Slot(str)
    def status(self, msg) -> None:
        self._status_text.setText(msg)
