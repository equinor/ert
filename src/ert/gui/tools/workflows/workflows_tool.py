from qtpy.QtGui import QIcon

from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.workflows import RunWorkflowWidget


class WorkflowsTool(Tool):
    def __init__(self, config, notifier):
        self.notifier = notifier
        self.config = config
        enabled = len(config.workflows) > 0
        super().__init__(
            "Run workflow",
            QIcon("img:playlist_play.svg"),
            enabled,
        )

    def trigger(self):
        run_workflow_widget = RunWorkflowWidget(self.config, self.notifier)
        dialog = ClosableDialog("Run workflow", run_workflow_widget, self.parent())
        dialog.exec_()
        self.notifier.emitErtChange()  # workflow may have added new cases.
