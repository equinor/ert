from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtGui import QIcon

from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.workflows import RunWorkflowWidget

if TYPE_CHECKING:
    from ert.enkf_main import EnKFMain
    from ert.gui.ertnotifier import ErtNotifier


class WorkflowsTool(Tool):
    def __init__(self, ert: EnKFMain, notifier: ErtNotifier) -> None:
        self.notifier = notifier
        self.ert = ert
        enabled = len(ert.ert_config.workflows) > 0
        super().__init__(
            "Run workflow",
            QIcon("img:playlist_play.svg"),
            enabled,
        )

    def trigger(self) -> None:
        run_workflow_widget = RunWorkflowWidget(self.ert, self.notifier)
        dialog = ClosableDialog("Run workflow", run_workflow_widget, self.parent())  # type: ignore
        dialog.exec_()
        self.notifier.emitErtChange()  # workflow may have added new cases.
