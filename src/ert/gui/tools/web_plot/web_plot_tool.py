from qtpy.QtGui import QIcon

from ert.gui.tools import Tool

from .web_plot_window import WebPlotWindow


class WebPlotTool(Tool):
    def __init__(self, main_window, ens_path):
        super().__init__("Create WebPlot", QIcon("img:timeline.svg"))
        self.main_window = main_window
        self.ens_path = ens_path

    def trigger(self):
        plot_window = WebPlotWindow(self.main_window, ens_path=self.ens_path)
        plot_window.show()
