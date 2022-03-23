from qtpy.QtCore import QObject, Signal, Slot


class ErtNotifier(QObject):
    ertChanged = Signal()

    def __init__(self, config_file: str):
        QObject.__init__(self)
        self._config_file = config_file

    @property
    def config_file(self):
        """@rtype: str"""
        return self._config_file

    @Slot()
    def emitErtChange(self):
        self.ertChanged.emit()
