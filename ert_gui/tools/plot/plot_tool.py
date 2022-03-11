from ert_gui.ertwidgets import resourceIcon
from ert_gui.tools import Tool
from ert_gui.tools.plot import PlotWindow


class PlotTool(Tool):
    def __init__(self, ert, config_file):
        self.ert = ert
        super(PlotTool, self).__init__(
            "Create Plot", "tools/plot", resourceIcon("ide/chart_curve_add")
        )
        self._config_file = config_file

    def trigger(self):
        plot_window = PlotWindow(self.ert, self._config_file, self.parent())
        plot_window.show()
