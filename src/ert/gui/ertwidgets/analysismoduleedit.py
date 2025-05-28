from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QMargins, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QToolButton, QWidget

from ert.gui.ertwidgets import ClosableDialog
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel

if TYPE_CHECKING:
    from ert.config import AnalysisModule


class AnalysisModuleEdit(QWidget):
    def __init__(
        self,
        analysis_module: AnalysisModule,
        ensemble_size: int,
    ) -> None:
        self.analysis_module = analysis_module
        self.ensemble_size = ensemble_size
        QWidget.__init__(self)

        layout = QHBoxLayout()

        variables_popup_button = QToolButton()
        variables_popup_button.setIcon(QIcon("img:edit.svg"))
        variables_popup_button.clicked.connect(self.showVariablesPopup)
        variables_popup_button.setMaximumSize(20, 20)

        layout.addWidget(variables_popup_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addStretch()

        self.setLayout(layout)

    def showVariablesPopup(self) -> None:
        variable_dialog = AnalysisModuleVariablesPanel(
            self.analysis_module, self.ensemble_size
        )
        dialog = ClosableDialog(
            "Edit variables",
            variable_dialog,
            self.parent(),  # type: ignore
        )
        dialog.exec()
