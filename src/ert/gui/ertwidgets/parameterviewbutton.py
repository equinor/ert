from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ert.gui.ertwidgets import ClosableDialog
from ert.gui.ertwidgets.parameterviewer import ParametersViewerPanel

if TYPE_CHECKING:
    from ert.config import EnsembleConfig


class ParameterViewButton(QWidget):
    def __init__(
        self,
        ensemble_config: EnsembleConfig,
    ) -> None:
        self.ensemble_config = ensemble_config
        QWidget.__init__(self)

        layout = QHBoxLayout()

        parameter_viewer_button = QPushButton("Show parameters")
        parameter_viewer_button.setMinimumWidth(50)
        parameter_viewer_button.clicked.connect(self.show_parameter_viewer)

        layout.addWidget(parameter_viewer_button)
        layout.addStretch()

        self.setLayout(layout)

    def show_parameter_viewer(self) -> None:
        parameter_dialog = ParametersViewerPanel(self.ensemble_config)
        dialog = ClosableDialog(
            "View parameters",
            parameter_dialog,
            self.parent(),  # type: ignore
        )
        dialog.exec()
