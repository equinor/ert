from ert_gui.ertwidgets import resourceIcon
from ert_gui.tools import Tool
from ert_gui.tools.plot import PlotWindow


class PlotTool(Tool):
    def __init__(self, args, storage_client):
        super(PlotTool, self).__init__(
            "Create Plot", "tools/plot", resourceIcon("ide/chart_curve_add")
        )
        self._args = args
        self._storage_client = storage_client

    def trigger(self):
        plot_window = PlotWindow(self._args, self.parent(), self._storage_client)
        plot_window.show()
