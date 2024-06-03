from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon

from ert.gui.tools import Tool
from ert.gui.tools.manage_experiments.ensemble_init_configuration import (
    EnsembleInitializationConfigurationPanel,
)

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier


class ManageExperimentsTool(Tool):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier, ensemble_size: int):
        super().__init__("Manage experiments", QIcon("img:build_wrench.svg"))
        self.notifier = notifier
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self._ensemble_management_widget = None

    def trigger(self):
        if not self._ensemble_management_widget:
            self._ensemble_management_widget = EnsembleInitializationConfigurationPanel(
                self.ert_config, self.notifier, self.ensemble_size
            )
            self._ensemble_management_widget.setWindowModality(
                Qt.WindowModality.ApplicationModal
            )

        self._ensemble_management_widget.show()
