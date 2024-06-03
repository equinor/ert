from typing import Optional

from qtpy.QtCore import QObject, Signal, Slot


class ValueModel(QObject):
    valueChanged = Signal(str)

    def __init__(self, value: Optional[str] = ""):
        super().__init__()
        self._value = value

    def getValue(self) -> Optional[str]:
        return self._value

    @Slot(str)
    def setValue(self, value: Optional[str]) -> None:
        self._value = value
        self.valueChanged.emit(value)

    def __repr__(self) -> str:
        return f'ValueModel(QObject)(value = "{self._value}")'
