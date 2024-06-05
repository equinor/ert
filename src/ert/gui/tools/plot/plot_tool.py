from typing import Optional

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QWidget

from ert.gui.tools import Tool

from .plot_window import PlotWindow


class PlotTool(Tool):
    def __init__(self, config_file: str, main_window: Optional[QWidget]):
        super().__init__("Create plot", QIcon("img:timeline.svg"))
        self._config_file = config_file
        self.main_window = main_window

    def trigger(self) -> None:
        plot_window = PlotWindow(self._config_file, self.main_window)
        plot_window.show()
