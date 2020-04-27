from ert_gui.ertwidgets import resourceIcon
from ert_gui.tools import Tool
from ert_gui.tools.plot import PlotWindow


class PlotTool(Tool):
    def __init__(self, config_file, storage_client):
        super(PlotTool, self).__init__(
            "Create Plot", "tools/plot", resourceIcon("ide/chart_curve_add")
        )
        self._config_file = config_file
        self._storage_client = storage_client

    def trigger(self):
        plot_window = PlotWindow(self._config_file, self._storage_client, self.parent())
        plot_window.show()
