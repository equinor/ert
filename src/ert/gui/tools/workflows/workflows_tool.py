from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QIcon

from ert.gui.ertwidgets import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.workflows import RunWorkflowWidget

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier


class WorkflowsTool(Tool):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier) -> None:
        self.notifier = notifier
        self.config = config
        enabled = len(config.workflows) > 0
        super().__init__(
            "Run workflow",
            QIcon("img:playlist_play.svg"),
            enabled,
        )

    def trigger(self) -> None:
        run_workflow_widget = RunWorkflowWidget(self.config, self.notifier)
        dialog = ClosableDialog("Run workflow", run_workflow_widget, self.parent())  # type: ignore
        dialog.exec()
        self.notifier.emitErtChange()  # workflow may have added new cases.
