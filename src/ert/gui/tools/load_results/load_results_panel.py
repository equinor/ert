from PyQt5.QtWidgets import QMessageBox
from qtpy.QtWidgets import QComboBox, QFormLayout, QTextEdit, QWidget

from ert.gui.ertwidgets.message_box import ErtMessageBox
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.all_cases_model import AllCasesModel
from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.libres_facade import LibresFacade
from ert.shared.ide.keywords.definitions import IntegerArgument, RangeStringArgument
from ert.shared.models.base_run_model import _LogAggregration, captured_logs


class LoadResultsPanel(QWidget):
    def __init__(self, facade: LibresFacade):
        self.facade = facade
        QWidget.__init__(self)

        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        self._dynamic = False

        self.setWindowTitle("Load results manually")
        self.activateWindow()

        layout = QFormLayout()
        current_case = facade.get_current_case_name()

        run_path_text = QTextEdit()
        run_path_text.setText(self.readCurrentRunPath())
        run_path_text.setDisabled(True)
        run_path_text.setFixedHeight(80)

        layout.addRow("Load data from current run path: ", run_path_text)

        self._case_model = AllCasesModel(self.facade)
        self._case_combo = QComboBox()
        self._case_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self._case_combo.setMinimumContentsLength(20)
        self._case_combo.setModel(self._case_model)
        self._case_combo.setCurrentIndex(self._case_model.indexOf(current_case))
        layout.addRow("Load into case:", self._case_combo)

        self._active_realizations_model = ActiveRealizationsModel(self.facade)
        self._active_realizations_field = StringBox(
            self._active_realizations_model, "load_results_manually/Realizations"
        )
        self._active_realizations_field.setValidator(RangeStringArgument())
        layout.addRow("Realizations to load:", self._active_realizations_field)

        self._iterations_model = ValueModel(self.facade.get_number_of_iterations())
        self._iterations_field = StringBox(
            self._iterations_model, "load_results_manually/iterations"
        )
        self._iterations_field.setValidator(IntegerArgument())
        layout.addRow("Iteration to load:", self._iterations_field)

        self.setLayout(layout)

    def readCurrentRunPath(self):
        current_case = self.facade.get_current_case_name()
        run_path = self.facade.run_path
        run_path = run_path.replace("<ERTCASE>", current_case)
        run_path = run_path.replace("<ERT-CASE>", current_case)
        return run_path

    def load(self):
        all_cases = self._case_model.getAllItems()
        selected_case = all_cases[self._case_combo.currentIndex()]
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
            loaded = self.facade.load_from_forward_model(
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

    def setCurrectCase(self):
        current_case = self.facade.get_current_case_name()
        self._case_combo.setCurrentIndex(self._case_model.indexOf(current_case))
