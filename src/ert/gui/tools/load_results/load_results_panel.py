from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout, QMessageBox, QWidget

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    EnsembleSelector,
    ErtMessageBox,
    MultiLineStringBox,
    QApplication,
    StringBox,
    ValueModel,
)
from ert.libres_facade import LibresFacade
from ert.run_models.base_run_model import captured_logs
from ert.validation import IntegerArgument, RangeStringArgument, RunPathArgument


class LoadResultsPanel(QWidget):
    panelConfigurationChanged = Signal()

    def __init__(self, facade: LibresFacade, notifier: ErtNotifier):
        self._facade = facade
        QWidget.__init__(self)

        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        self._dynamic = False
        self._notifier = notifier

        self.setWindowTitle("Load results manually")
        self.activateWindow()

        layout = QFormLayout()

        self.run_path_text_model = ValueModel()
        self._run_path_field = MultiLineStringBox(
            self.run_path_text_model,  # type: ignore
            default_string="",
            readonly=True,
        )
        self._run_path_field.setValidator(RunPathArgument())
        self._run_path_field.setObjectName("run_path_field_lrm")
        self._run_path_field.setFixedHeight(80)
        self._run_path_field.setText(self.readCurrentRunPath())

        layout.addRow("Load data from current run path: ", self._run_path_field)
        ensemble_selector = EnsembleSelector(self._notifier)
        layout.addRow("Load into ensemble:", ensemble_selector)
        self._ensemble_selector = ensemble_selector

        self._active_realizations_model = ActiveRealizationsModel(
            self._facade.get_ensemble_size()
        )
        self._active_realizations_field = StringBox(
            self._active_realizations_model,  # type: ignore
            "load_results_manually/Realizations",
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(self._facade.get_ensemble_size()),
        )
        self._active_realizations_field.setObjectName("active_realizations_lrm")
        layout.addRow("Realizations to load:", self._active_realizations_field)

        self._iterations_model = ValueModel(0)  # type: ignore
        self._iterations_field = StringBox(
            self._iterations_model,  # type: ignore
            "load_results_manually/iterations",
        )
        self._iterations_field.setValidator(IntegerArgument(from_value=0))
        self._iterations_field.setObjectName("iterations_field_lrm")
        layout.addRow("Iteration to load:", self._iterations_field)
        self._run_path_field.getValidationSupport().validationChanged.connect(
            self.panelConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.panelConfigurationChanged
        )
        self._iterations_field.getValidationSupport().validationChanged.connect(
            self.panelConfigurationChanged
        )

        self.setLayout(layout)

    def refresh(self) -> None:
        self._run_path_field.refresh()

    def readCurrentRunPath(self) -> str:
        current_ensemble = self._notifier.current_ensemble_name
        run_path = self._facade.run_path
        run_path = run_path.replace("<ERTCASE>", current_ensemble)
        run_path = run_path.replace("<ERT-CASE>", current_ensemble)
        return run_path

    def isConfigurationValid(self) -> bool:
        return (
            self._run_path_field.isValid()
            and self._active_realizations_field.isValid()
            and self._iterations_field.isValid()
        )

    def load(self) -> int:
        selected_ensemble = self._notifier.current_ensemble
        realizations = self._active_realizations_model.getActiveRealizationsMask()
        iteration = self._iterations_model.getValue()
        try:
            if iteration is None:
                iteration = ""
            iteration_int = int(iteration)
        except ValueError:
            QMessageBox.warning(
                self,
                "Warning",
                (
                    "Expected an integer number in iteration field, "
                    f'got "{iteration}"'
                ),
            )
            return False

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        messages: list[str] = []
        with captured_logs(messages):
            loaded = self._facade.load_from_forward_model(
                selected_ensemble,  # type: ignore
                realizations,  # type: ignore
                iteration_int,
            )
        QApplication.restoreOverrideCursor()

        if loaded == realizations.count(True):
            QMessageBox.information(
                self, "Success", "Successfully loaded all realisations"
            )
        elif loaded > 0:
            msg = ErtMessageBox(
                f"Successfully loaded {loaded} realizations", "\n".join(messages)
            )
            msg.exec_()
        else:
            msg = ErtMessageBox("No realizations loaded", "\n".join(messages))
            msg.exec_()
        return loaded
