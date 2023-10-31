from qtpy.QtGui import QIcon

from ert.gui.tools import Tool

from .plot_window import PlotWindow


class PlotTool(Tool):
    def __init__(self, config_file, main_window):
        super().__init__("Create plot", QIcon("img:timeline.svg"))
        self._config_file = config_file
        self.main_window = main_window

    def trigger(self):
        plot_window = PlotWindow(self._config_file, self.main_window)
        plot_window.show()
