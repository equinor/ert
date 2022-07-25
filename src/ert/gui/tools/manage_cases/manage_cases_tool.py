from ert.gui.ertwidgets import resourceIcon
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)


class ManageCasesTool(Tool):
    def __init__(self, ert, notifier):
        self.notifier = notifier
        self.ert = ert
        super().__init__(
            "Manage cases", "tools/manage_cases", resourceIcon("build_wrench.svg")
        )

    def trigger(self):
        case_management_widget = CaseInitializationConfigurationPanel(
            self.ert, self.notifier
        )

        dialog = ClosableDialog("Manage cases", case_management_widget, self.parent())
        dialog.exec_()
