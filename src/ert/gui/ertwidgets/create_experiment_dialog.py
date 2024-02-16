from qtpy.QtCore import (
    Qt,
    Signal,
)
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
)


class CreateExperimentDialog(QDialog):
    onDone = Signal(str, str)

    def __init__(
        self,
        title: str = "Create new experiment",
        parent=None,
    ):
        QDialog.__init__(self, parent=parent)
        self.setModal(True)
        self.setWindowTitle(title)
        self.setFixedSize(400, 100)

        layout = QGridLayout()

        experiment_label = QLabel("Experiment name:")
        self._experiment_edit = QLineEdit()
        self._experiment_edit.setPlaceholderText("My experiment")

        ensemble_label = QLabel("Ensemble name:")
        self._ensemble_edit = QLineEdit()
        self._ensemble_edit.setPlaceholderText("My ensemble")

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._ok_button = buttons.button(QDialogButtonBox.Ok)
        assert self._ok_button

        self._ok_button.clicked.connect(
            lambda: self.onDone.emit(
                self._experiment_edit.text(), self._ensemble_edit.text()
            )
        )
        self._ok_button.setEnabled(False)

        def enableOkButton():
            self._ok_button.setEnabled(
                len(self._experiment_edit.text()) != 0
                and len(self._ensemble_edit.text()) != 0
            )

        self._experiment_edit.textChanged.connect(enableOkButton)
        self._ensemble_edit.textChanged.connect(enableOkButton)

        layout.addWidget(experiment_label, 0, 0)
        layout.addWidget(self._experiment_edit, 0, 1)
        layout.addWidget(ensemble_label, 1, 0)
        layout.addWidget(self._ensemble_edit, 1, 1)
        layout.addWidget(buttons, 2, 1)

        self.setLayout(layout)

        self._experiment_edit.setFocus()

    @property
    def experiment_name(self) -> str:
        return self._experiment_edit.text()

    @property
    def ensemble_name(self) -> str:
        return self._ensemble_edit.text()
