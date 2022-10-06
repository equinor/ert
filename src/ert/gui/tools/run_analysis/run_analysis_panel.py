from qtpy.QtWidgets import QFormLayout, QLineEdit, QWidget

from ert.gui.ertwidgets.analysismoduleedit import AnalysisModuleEdit
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.libres_facade import LibresFacade


class RunAnalysisPanel(QWidget):
    def __init__(self, ert, notifier):
        self.ert = ert
        QWidget.__init__(self)

        self.setWindowTitle("Run analysis")
        self.activateWindow()

        self.analysis_module = AnalysisModuleEdit(
            LibresFacade(ert),
            help_link="config/analysis/analysis_module",
        )
        self.target_case_text = QLineEdit()
        self.source_case_selector = CaseSelector(
            LibresFacade(self.ert), notifier, update_ert=False
        )

        layout = QFormLayout()
        layout.addRow("Analysis", self.analysis_module)
        layout.addRow("Target case", self.target_case_text)
        layout.addRow("Source case", self.source_case_selector)
        self.setLayout(layout)

    def target_case(self):
        return str(self.target_case_text.text())

    def source_case(self):
        return str(self.source_case_selector.currentText())
