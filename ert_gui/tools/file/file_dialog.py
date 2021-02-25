from qtpy.QtCore import QThread, Slot
from qtpy.QtWidgets import QDialog, QMessageBox, QDialogButtonBox, QVBoxLayout

from .file_model import FileModel
from .file_view import FileView
from .file_update_worker import FileUpdateWorker


class FileDialog(QDialog):
    def __init__(
        self, file_name, job_name, job_number, realization, iteration, parent=None
    ):
        super(FileDialog, self).__init__(parent)

        self.setWindowTitle(
            "{} # {} Realization: {} Iteration: {}".format(
                job_name, job_number, realization, iteration
            )
        )

        self._file_name = file_name
        try:
            self._file = open(file_name, "r")
        except OSError as error:
            self._mb = QMessageBox(
                QMessageBox.Critical,
                "Error opening file",
                error.strerror,
                QMessageBox.Ok,
                self,
            )
            self._mb.finished.connect(self.accept)
            self._mb.show()
            return

        self._model = FileModel()
        self._view = FileView()
        self._view.setModel(self._model)

        self._init_layout()
        self._init_thread()

        self.show()

    @Slot()
    def _stop_thread(self):
        self._thread.quit()
        self._thread.wait()

    def _init_layout(self):
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        self._follow = dialog_buttons.addButton("Follow", QDialogButtonBox.ActionRole)
        self._copy_all = dialog_buttons.addButton(
            "Copy All", QDialogButtonBox.ActionRole
        )
        dialog_buttons.accepted.connect(self.accept)

        self._follow.setCheckable(True)
        self._follow.toggled.connect(self._view.enable_follow_mode)
        self._copy_all.clicked.connect(self._model.copy_all)

        layout = QVBoxLayout(self)
        layout.addWidget(self._view)
        layout.addWidget(dialog_buttons)

    def _init_thread(self):
        self._thread = QThread()

        self._worker = FileUpdateWorker(self._file)
        self._worker.moveToThread(self._thread)
        self._worker.read.connect(self._model.append_text)

        self._thread.started.connect(self._worker.setup)
        self._thread.finished.connect(self._worker.stop)
        self._thread.finished.connect(self._worker.deleteLater)
        self.finished.connect(self._stop_thread)
        self._thread.start()
