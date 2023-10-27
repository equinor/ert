from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtGui import QIcon

from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)

if TYPE_CHECKING:
    from ert.config import ErtConfig


class ManageCasesTool(Tool):
    def __init__(self, config: ErtConfig, notifier, ensemble_size: int):
        self.notifier = notifier
        self.ert_config = config
        self.ensemble_size = ensemble_size
        super().__init__("Manage cases", QIcon("img:build_wrench.svg"))

    def trigger(self):
        case_management_widget = CaseInitializationConfigurationPanel(
            self.ert_config, self.notifier, self.ensemble_size
        )

        dialog = ClosableDialog("Manage cases", case_management_widget, self.parent())
        dialog.setObjectName("manage-cases")
        dialog.exec_()
