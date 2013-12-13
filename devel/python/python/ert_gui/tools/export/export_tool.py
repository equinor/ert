from ert_gui.tools import Tool
from ert_gui.widgets import util


class ExportTool(Tool):
    def __init__(self):
        super(ExportTool, self).__init__("Export Data", util.resourceIcon("ide/table_export"), enabled=False)

    def trigger(self):
        pass

