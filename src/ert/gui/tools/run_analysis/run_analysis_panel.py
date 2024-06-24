from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import QFormLayout, QLineEdit, QWidget

from ert.config import AnalysisModule
from ert.gui.ertwidgets.analysismoduleedit import AnalysisModuleEdit
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector

if TYPE_CHECKING:
    from ert.gui.ertnotifier import ErtNotifier


class RunAnalysisPanel(QWidget):
    def __init__(
        self, analysis_module: AnalysisModule, ensemble_size: int, notifier: ErtNotifier
    ) -> None:
        QWidget.__init__(self)

        self.setWindowTitle("Run analysis")
        self.activateWindow()
        self.analysis_module = AnalysisModuleEdit(analysis_module, ensemble_size)
        self.target_ensemble_text = QLineEdit()
        self.source_ensemble_selector = EnsembleSelector(notifier, update_ert=False)

        self.setMinimumSize(400, 100)

        layout = QFormLayout()
        layout.addRow("Analysis", self.analysis_module)
        layout.addRow("Target ensemble", self.target_ensemble_text)
        layout.addRow("Source ensemble", self.source_ensemble_selector)
        self.setLayout(layout)

    def target_ensemble(self) -> str:
        return str(self.target_ensemble_text.text())

    def source_ensemble(self) -> Any:
        return self.source_ensemble_selector.currentData()
