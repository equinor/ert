from ert.gui.ertwidgets import resourceIcon
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.tools import Tool
from ert.gui.tools.workflows import RunWorkflowWidget


class WorkflowsTool(Tool):
    def __init__(self, ert, notifier):
        self.notifier = notifier
        self.ert = ert
        enabled = len(ert.getWorkflowList().getWorkflowNames()) > 0
        super().__init__(
            "Run workflow",
            "tools/workflows",
            resourceIcon("playlist_play.svg"),
            enabled,
        )

    def trigger(self):
        run_workflow_widget = RunWorkflowWidget(self.ert)
        dialog = ClosableDialog("Run workflow", run_workflow_widget, self.parent())
        dialog.exec_()
        self.notifier.emitErtChange()  # workflow may have added new cases.
