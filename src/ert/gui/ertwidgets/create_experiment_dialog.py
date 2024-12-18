from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QWidget,
)

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import StringBox, TextModel, ValueModel
from ert.validation import ExperimentValidation, IntegerArgument, ProperNameArgument


class CreateExperimentDialog(QDialog):
    onDone = Signal(str, str, int)

    def __init__(
        self,
        notifier: ErtNotifier,
        title: str = "Create new experiment",
        parent: QWidget | None = None,
    ) -> None:
        QDialog.__init__(self, parent=parent)
        self.setModal(True)
        self.setWindowTitle(title)
        self.setFixedSize(450, 120)

        layout = QGridLayout()

        experiment_label = QLabel("Experiment name:")
        self._experiment_edit = StringBox(
            TextModel(""),
            placeholder_text=notifier.storage.get_unique_experiment_name(
                "new_experiment"
            ),
            minimum_width=200,
        )
        self._experiment_edit.setValidator(ExperimentValidation(notifier.storage))

        ensemble_label = QLabel("Ensemble name:")
        self._ensemble_edit = StringBox(
            TextModel(""),
            placeholder_text=notifier.storage.get_unique_experiment_name("ensemble"),
            minimum_width=200,
        )
        self._ensemble_edit.setValidator(ProperNameArgument())

        iteration_label = QLabel("Ensemble iteration:")
        self._iterations_model = ValueModel(0)  # type: ignore
        self._iterations_field = StringBox(
            self._iterations_model,  # type: ignore
            "0",
            minimum_width=200,
        )
        self._iterations_field.setValidator(IntegerArgument(from_value=0))
        self._iterations_field.setObjectName("iterations_field_ced")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert ok_button
        self._ok_button = ok_button

        self._ok_button.clicked.connect(
            lambda: self.onDone.emit(
                self.experiment_name, self.ensemble_name, self.iteration
            )
        )

        def enableOkButton() -> None:
            self._ok_button.setEnabled(self.isConfigurationValid())

        self._experiment_edit.textChanged.connect(enableOkButton)
        self._ensemble_edit.textChanged.connect(enableOkButton)
        self._iterations_field.textChanged.connect(enableOkButton)

        layout.addWidget(experiment_label, 0, 0)
        layout.addWidget(self._experiment_edit, 0, 1)
        layout.addWidget(ensemble_label, 1, 0)
        layout.addWidget(self._ensemble_edit, 1, 1)
        layout.addWidget(iteration_label, 2, 0)
        layout.addWidget(self._iterations_field, 2, 1)
        layout.addWidget(buttons, 3, 1)

        self.setLayout(layout)

        self._experiment_edit.getValidationSupport().validationChanged.connect(
            enableOkButton
        )

        self._ensemble_edit.getValidationSupport().validationChanged.connect(
            enableOkButton
        )

        self._experiment_edit.setFocus()

    @property
    def experiment_name(self) -> str:
        return self._experiment_edit.get_text

    @property
    def ensemble_name(self) -> str:
        return self._ensemble_edit.get_text

    @property
    def iteration(self) -> int:
        return int(self._iterations_field.get_text)

    def isConfigurationValid(self) -> bool:
        return (
            self._experiment_edit.isValid()
            and self._ensemble_edit.isValid()
            and self._iterations_field.isValid()
        )
