from __future__ import annotations

from PyQt6.QtCore import QMargins, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ert.config import AnalysisConfig
from ert.gui.icon_utils import load_icon

from .analysismodulevariablespanel import AnalysisModuleVariablesPanel
from .closabledialog import ClosableDialog


class AnalysisModuleEdit(QWidget):
    on_dialog_closed = Signal(dict)

    def __init__(
        self,
        analysis_config: AnalysisConfig,
        ensemble_size: int,
    ) -> None:
        QWidget.__init__(self)

        self.analysis_config = analysis_config
        self.ensemble_size = ensemble_size

        layout = QHBoxLayout()

        variables_popup_button = QPushButton("Edit")
        variables_popup_button.setObjectName("analysis_variables_popup_button")
        variables_popup_button.setIcon(load_icon("edit.svg"))
        variables_popup_button.clicked.connect(self.showVariablesPopup)

        layout.addWidget(variables_popup_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addStretch()

        self.setLayout(layout)

    def showVariablesPopup(self) -> None:
        variable_dialog = AnalysisModuleVariablesPanel(
            self.analysis_config,
            self.ensemble_size,
        )
        dialog = ClosableDialog(
            "Edit variables",
            variable_dialog,
            self.parent(),  # type: ignore
        )
        dialog.finished.connect(
            lambda _: self.on_dialog_closed.emit(
                variable_dialog.changed_updated_parameter_strategies
            )
        )
        dialog.exec()
