from __future__ import annotations

from qtpy.QtWidgets import QFormLayout, QMessageBox, QTextEdit, QWidget

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.ertwidgets.message_box import ErtMessageBox
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.libres_facade import LibresFacade
from ert.run_models.base_run_model import captured_logs
from ert.validation import IntegerArgument, RangeStringArgument


class LoadResultsPanel(QWidget):
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

        run_path_text = QTextEdit()
        run_path_text.setText(self.readCurrentRunPath())
        run_path_text.setDisabled(True)
        run_path_text.setFixedHeight(80)

        layout.addRow("Load data from current run path: ", run_path_text)

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
        self._active_realizations_field.setValidator(RangeStringArgument())
        layout.addRow("Realizations to load:", self._active_realizations_field)

        self._iterations_model = ValueModel(0)  # type: ignore
        self._iterations_field = StringBox(
            self._iterations_model,  # type: ignore
            "load_results_manually/iterations",
        )
        self._iterations_field.setValidator(IntegerArgument(from_value=0))
        layout.addRow("Iteration to load:", self._iterations_field)

        self.setLayout(layout)

    def readCurrentRunPath(self) -> str:
        current_ensemble = self._notifier.current_ensemble_name
        run_path = self._facade.run_path
        run_path = run_path.replace("<ERTCASE>", current_ensemble)
        run_path = run_path.replace("<ERT-CASE>", current_ensemble)
        return run_path

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
        messages: list[str] = []
        with captured_logs(messages):
            loaded = self._facade.load_from_forward_model(
                selected_ensemble,  # type: ignore
                realizations,  # type: ignore
                iteration_int,
            )

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
