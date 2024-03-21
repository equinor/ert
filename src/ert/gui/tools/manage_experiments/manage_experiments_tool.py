from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtGui import QIcon

from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.manage_experiments.ensemble_init_configuration import (
    EnsembleInitializationConfigurationPanel,
)

if TYPE_CHECKING:
    from ert.config import ErtConfig


class ManageExperimentsTool(Tool):
    def __init__(self, config: ErtConfig, notifier, ensemble_size: int):
        self.notifier = notifier
        self.ert_config = config
        self.ensemble_size = ensemble_size
        super().__init__("Manage experiments", QIcon("img:build_wrench.svg"))

    def trigger(self):
        ensemble_management_widget = EnsembleInitializationConfigurationPanel(
            self.ert_config, self.notifier, self.ensemble_size
        )

        dialog = ClosableDialog(
            "Manage experiments", ensemble_management_widget, self.parent()
        )
        dialog.setObjectName("manage-experiments")
        dialog.exec_()
