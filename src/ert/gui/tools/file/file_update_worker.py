from qtpy.QtCore import QObject, QTimer, Signal, Slot


class FileUpdateWorker(QObject):
    POLL_TIMER_MS = 1000
    POLL_BUFFER_SIZE = 2**16  # 16KiB
    INIT_BUFFER_SIZE = 2**20  # 1MiB

    read = Signal(str)

    def __init__(self, file, parent=None):
        super().__init__(parent)
        self._file = file
        self._timer = None

    @Slot()
    def stop(self):
        self._file.close()
        self._timer.stop()

    @Slot()
    def setup(self):
        text = self._file.read(self.INIT_BUFFER_SIZE)
        self._send_text(text)

        self._timer = QTimer()
        self._timer.timeout.connect(self._poll_file)
        self._timer.start(self.POLL_TIMER_MS)

    @Slot()
    def _poll_file(self):
        text = self._file.read(self.POLL_BUFFER_SIZE)
        self._send_text(text)

    def _send_text(self, text):
        if len(text) > 0:
            self.read.emit(text)
