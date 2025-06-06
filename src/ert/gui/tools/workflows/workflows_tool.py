from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QIcon

from ert.gui.ertwidgets import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.workflows import RunWorkflowWidget

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier
import logging

logger = logging.getLogger(__name__)


class WorkflowsTool(Tool):
    def __init__(
        self, config: ErtConfig, notifier: ErtNotifier, trigger_source: str = ""
    ) -> None:
        self.trigger_source = trigger_source
        self.notifier = notifier
        self.config = config
        enabled = len(config.workflows) > 0
        super().__init__(
            "Run workflow",
            QIcon("img:playlist_play.svg"),
            enabled,
        )

    def trigger(self) -> None:
        logger.info(
            f"WorkflowsTool triggered from {self.trigger_source}"
            if self.trigger_source
            else ""
        )
        run_workflow_widget = RunWorkflowWidget(self.config, self.notifier)
        dialog = ClosableDialog("Run workflow", run_workflow_widget, self.parent())  # type: ignore
        dialog.exec()
        self.notifier.emitErtChange()  # workflow may have added new cases.
