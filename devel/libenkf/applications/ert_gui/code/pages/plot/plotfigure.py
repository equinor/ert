from matplotlib.figure import Figure
import matplotlib.lines
import matplotlib.text

import numpy
import erttypes
import pages.plot.plotsettings
from pages.plot.plotrenderer import DefaultPlotRenderer

class PlotFigure:
    """A simple wrapper for a matplotlib Figure. Associated with a specified renderer."""
    def __init__(self):
        self.fig = Figure(dpi=100)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.15)
        self.axes.set_xlim()
        self.plot_renderer = DefaultPlotRenderer()

    def getFigure(self):
        """Return the matplotlib figure."""
        return self.fig

    def drawPlot(self, plot_data, plot_settings):
        """Draw a plot based on the specified data and settings."""
        self.axes.cla()

        self.plot_renderer.drawPlot(self.axes, plot_data, plot_settings)

        if plot_data.getXDataType() == "time":
            self.fig.autofmt_xdate()



