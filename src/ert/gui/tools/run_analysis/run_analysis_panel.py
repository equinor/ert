from qtpy.QtWidgets import QFormLayout, QLineEdit, QWidget

from ert.config import AnalysisModule
from ert.gui.ertwidgets.analysismoduleedit import AnalysisModuleEdit
from ert.gui.ertwidgets.caseselector import CaseSelector


class RunAnalysisPanel(QWidget):
    def __init__(self, analysis_module: AnalysisModule, ensemble_size: int, notifier):
        QWidget.__init__(self)

        self.setWindowTitle("Run analysis")
        self.activateWindow()
        self.analysis_module = AnalysisModuleEdit(analysis_module, ensemble_size)
        self.target_case_text = QLineEdit()
        self.source_case_selector = CaseSelector(notifier, update_ert=False)

        layout = QFormLayout()
        layout.addRow("Analysis", self.analysis_module)
        layout.addRow("Target case", self.target_case_text)
        layout.addRow("Source case", self.source_case_selector)
        self.setLayout(layout)

    def target_case(self):
        return str(self.target_case_text.text())

    def source_case(self):
        return self.source_case_selector.currentData()
