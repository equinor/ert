from qtpy.QtCore import QObject, Signal, Slot


class ValueModel(QObject):
    valueChanged = Signal(str)

    def __init__(self, value=""):
        super().__init__()
        self._value = value

    def getValue(self):
        """@rtype: str"""
        return self._value

    @Slot(str)
    def setValue(self, value):
        self._value = value
        self.valueChanged.emit(value)

    def __repr__(self):
        return f'ValueModel(QObject)(value = "{self._value}")'
