from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot


class ValueModel(QObject):
    valueChanged = Signal(str)

    def __init__(self, value: str | None = "") -> None:
        super().__init__()
        self._value = value

    def getValue(self) -> str | None:
        return self._value

    @Slot(str)
    def setValue(self, value: str | None) -> None:
        self._value = value
        self.valueChanged.emit(value)

    def __repr__(self) -> str:
        return f'ValueModel(QObject)(value = "{self._value}")'
