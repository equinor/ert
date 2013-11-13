from ert_gui.tools import Tool
from ert_gui.widgets import util


class WorkflowsTool(Tool):
    def __init__(self):
        super(WorkflowsTool, self).__init__("Run Workflow", util.resourceIcon("ide/to_do_list_checked_1"), enabled=False)

    def trigger(self):
        pass

