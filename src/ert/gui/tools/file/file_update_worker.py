from io import TextIOWrapper

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot


class FileUpdateWorker(QObject):
    POLL_TIMER_MS = 1000
    POLL_BUFFER_SIZE = 2**16  # 16KiB
    INIT_BUFFER_SIZE = 2**20  # 1MiB

    read = Signal(str)

    def __init__(self, file: TextIOWrapper, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._file = file
        self._timer: QTimer | None = None

    @Slot()
    def stop(self) -> None:
        self._file.close()
        if self._timer is not None:
            self._timer.stop()

    @Slot()
    def setup(self) -> None:
        text = self._file.read(self.INIT_BUFFER_SIZE)
        self._send_text(text)

        self._timer = QTimer()
        self._timer.timeout.connect(self._poll_file)
        self._timer.start(self.POLL_TIMER_MS)

    @Slot()
    def _poll_file(self) -> None:
        text = self._file.read(self.POLL_BUFFER_SIZE)
        self._send_text(text)

    def _send_text(self, text: str) -> None:
        if len(text) > 0:
            self.read.emit(text)
