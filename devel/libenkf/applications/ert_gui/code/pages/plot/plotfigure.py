from matplotlib.figure import Figure
import matplotlib.lines
import matplotlib.text

from plotter import Plotter
import numpy
import erttypes
import pages.plot.plotsettings
from pages.plot.plotrenderer import PlotRenderer

class PlotFigure:
    def __init__(self):
        self.fig = Figure(dpi=100)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlim()
        self.plot_renderer = PlotRenderer()

    def getFigure(self):
        return self.fig

    def drawPlot(self, plot_data, plot_settings):
        self.axes.cla()

        self.plot_renderer.drawPlot(self.axes, plot_data, plot_settings)

        if plot_data.getXDataType() == "time":
            self.fig.autofmt_xdate()



