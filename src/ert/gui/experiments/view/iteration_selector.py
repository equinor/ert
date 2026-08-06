from PyQt6.QtWidgets import QComboBox, QWidget


class IterationSelector(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)

    def add_iteration(self, label: str, data: int) -> None:
        was_on_latest = self.currentIndex() == self.count() - 1
        self.addItem(label, data)
        if was_on_latest:
            self.setCurrentIndex(self.count() - 1)
