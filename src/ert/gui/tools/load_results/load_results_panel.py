from PyQt5.QtWidgets import QMessageBox
from qtpy.QtWidgets import QFormLayout, QTextEdit, QWidget

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.message_box import ErtMessageBox
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.libres_facade import LibresFacade
from ert.run_models.base_run_model import _LogAggregration, captured_logs
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

        case_selector = CaseSelector(self._notifier)
        layout.addRow("Load into case:", case_selector)
        self._case_selector = case_selector

        self._active_realizations_model = ActiveRealizationsModel(
            self._facade.get_ensemble_size()
        )
        self._active_realizations_field = StringBox(
            self._active_realizations_model, "load_results_manually/Realizations"
        )
        self._active_realizations_field.setValidator(RangeStringArgument())
        layout.addRow("Realizations to load:", self._active_realizations_field)

        self._iterations_model = ValueModel(0)
        self._iterations_field = StringBox(
            self._iterations_model, "load_results_manually/iterations"
        )
        self._iterations_field.setValidator(IntegerArgument(from_value=0))
        layout.addRow("Iteration to load:", self._iterations_field)

        self.setLayout(layout)

    def readCurrentRunPath(self):
        current_case = self._notifier.current_case_name
        run_path = self._facade.run_path
        run_path = run_path.replace("<ERTCASE>", current_case)
        run_path = run_path.replace("<ERT-CASE>", current_case)
        return run_path

    def load(self) -> int:
        selected_case = self._notifier.current_case
        realizations = self._active_realizations_model.getActiveRealizationsMask()
        iteration = self._iterations_model.getValue()
        try:
            if iteration is None:
                iteration = ""
            iteration = int(iteration)
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
        logs: _LogAggregration = _LogAggregration()
        with captured_logs() as logs:
            loaded = self._facade.load_from_forward_model(
                selected_case, realizations, iteration
            )

        if loaded == realizations.count(True):
            QMessageBox.information(
                self, "Success", "Successfully loaded all realisations"
            )
        elif loaded > 0:
            msg = ErtMessageBox(
                f"Successfully loaded {loaded} realizations", "\n".join(logs.messages)
            )
            msg.exec_()
        else:
            msg = ErtMessageBox("No realizations loaded", "\n".join(logs.messages))
            msg.exec_()
        return loaded
