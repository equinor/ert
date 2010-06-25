from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import datetime
import time
from erttypes import time_t

from widgets.util import print_timing
from  pages.plot.plotdata import PlotData
import widgets

from PyQt4.QtCore import SIGNAL
import os

from plotconfig import PlotConfig
from plotter import Plotter
from plotfigure import PlotFigure, matplotlib

from PyQt4.QtGui import QFrame, QInputDialog, QSizePolicy
from pages.plot.plotsettingsxml import PlotSettingsSaver, PlotSettingsLoader
from pages.plot.plotsettings import PlotSettings
from pages.plot.plotsettingsxml import PlotSettingsCopyDialog

class PlotView(QFrame):
    """PlotPanel shows available plot result files and displays them"""

    def __init__(self):
        """Create a PlotPanel"""
        QFrame.__init__(self)

        self.data = PlotData()
        self.data.x_data_type = "number"
        self.data.setValid(False)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.plot_figure = PlotFigure()
        self.plot_settings = PlotSettings()

        self.canvas = FigureCanvas(self.plot_figure.getFigure())
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.mouse_handler = MouseHandler(self)

    def configureLine(self, line, plot_config):
        line.set_color(plot_config.color)
        line.set_alpha(plot_config.alpha)
        line.set_zorder(plot_config.z_order)
        line.set_linestyle(plot_config.style)

    def toggleLine(self, line):
        gid = line.get_gid()
        if gid in self.plot_settings.selected_members:
            plot_config = self.plot_settings.plot_config
            self.plot_settings.unselectMember(gid)
        else:
            plot_config = self.plot_settings.selected_plot_config
            self.plot_settings.selectMember(gid)

        self.configureLine(line, plot_config)

        self.emit(SIGNAL('plotSelectionChanged(array)'), self.plot_settings.selected_members)
        self.canvas.draw()


    def updateLimits(self, draw=True):
        self.plot_figure.updateLimits(self.plot_settings, self.data)
        
        if draw:
            self.canvas.draw()


    @widgets.util.may_take_a_long_time
    def drawPlot(self):
        self.plot_figure.drawPlot(self.data, self.plot_settings)        
        self.canvas.draw()

    def resizeEvent(self, event):
        QFrame.resizeEvent(self, event)
        self.canvas.resize(event.size().width(), event.size().height())

    def addAnnotations(self, plot_config_loader):
        annotations = plot_config_loader.getAnnotations()
        if not annotations is None:
            self.plot_figure.clearAnnotations()
            for annotation in annotations:
                self.plot_figure.annotate(*annotation)

    def loadSettings(self, name):
        plot_config_loader = PlotSettingsLoader()
        plot_config_loader.load(name, self.plot_settings)
        self.addAnnotations(plot_config_loader)

        self.emit(SIGNAL('plotSettingsChanged(PlotSettings)'), self.plot_settings)

    def setData(self, data):
        if self.data.isValid():
            plot_config_saver = PlotSettingsSaver()
            annotations = self.plot_figure.getAnnotations()
            plot_config_saver.save(self.data.getSaveName(), self.plot_settings, annotations)

        self.data = data
        self.plot_settings.setXDataType(data.getXDataType())
        self.plot_settings.setYDataType(data.getYDataType())

        if self.data.isValid():
            self.loadSettings(self.data.getSaveName())
        else:
            self.emit(SIGNAL('plotSelectionChanged(array)'), self.plot_settings.selected_members)

    def setXViewFactors(self, xminf, xmaxf, draw=True):
        self.plot_figure.setXViewFactors(self.plot_settings, xminf, xmaxf, self.data.x_min, self.data.x_max)

        if draw:
            self.canvas.draw()

    def setYViewFactors(self, yminf, ymaxf, draw=True):
        self.plot_figure.setYViewFactors(self.plot_settings, yminf, ymaxf, self.data.y_min, self.data.y_max)

        if draw:
            self.canvas.draw()

    def save(self):
        plot_path = self.plot_settings.getPlotPath()
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        #todo: popup save dialog

        path = plot_path + "/" + self.data.getTitle()
        self.plot_figure.getFigure().savefig(path + ".png", dpi=300, format="png")
        self.plot_figure.getFigure().savefig(path + ".pdf", dpi=300, format="pdf")

    def copyPlotSettings(self):
        plot_config_loader = PlotSettingsLoader()
        plot_config_loader.copy(self.plot_settings)
        self.addAnnotations(plot_config_loader)

        self.emit(SIGNAL('plotSettingsChanged(PlotSettings)'), self.plot_settings)

    def setPlotPath(self, plot_path):
        self.plot_settings.setPlotPath(plot_path)

    def setPlotConfigPath(self, path):
        self.plot_settings.setPlotConfigPath(path)

    def __memberFinder(self, artist):
        return artist.get_gid() in self.plot_settings.selected_members

    def clearSelection(self):
        selected_lines = self.plot_figure.fig.findobj(self.__memberFinder)
        for line in selected_lines:
            self.configureLine(line, self.plot_settings.plot_config)
            self.plot_settings.unselectMember(line.get_gid())


        self.emit(SIGNAL('plotSelectionChanged(array)'), self.plot_settings.selected_members)
        self.canvas.draw()

    def displayToolTip(self, event):
        if not self.data is None and not event.xdata is None and not event.ydata is None:
            if self.data.getXDataType() == "time":
                date = matplotlib.dates.num2date(event.xdata)
                self.setToolTip("x: %s y: %04f" % (date.strftime("%d/%m-%Y"), event.ydata))
            else:
                self.setToolTip("x: %04f y: %04f" % (event.xdata, event.ydata))
        else:
            self.setToolTip("")

    def annotate(self, label, x, y, xt=None, yt=None):
        self.plot_figure.annotate(label, x, y, xt, yt)

    def removeAnnotation(self, annotation):
        self.plot_figure.removeAnnotation(annotation)

    def draw(self):
        self.canvas.draw()

    def setMinYLimit(self, value):
        self.plot_settings.setMinYLimit(value)
        self.updateLimits()

    def setMaxYLimit(self, value):
        self.plot_settings.setMaxYLimit(value)
        self.updateLimits()

    def setMinXLimit(self, value):
        self.plot_settings.setMinXLimit(value)
        self.updateLimits()

    def setMaxXLimit(self, value):
        self.plot_settings.setMaxXLimit(value)
        self.updateLimits()

    def getPlotConfigList(self):
        return self.plot_settings.getPlotConfigList()

class MouseHandler:

    def __init__(self, plot_view):
        self.plot_view = plot_view

        fig = plot_view.plot_figure.getFigure()
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('pick_event', self.on_pick)
        fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)

        self.button_position = None
        self.artist = None

    def on_press(self, event):
        if event.button == 3 and self.artist is None and not event.xdata is None and not event.ydata is None:
            label, success = QInputDialog.getText(self.plot_view, "New label", "Enter label:")

            if success and not str(label).strip() == "":
                self.plot_view.annotate(str(label), event.xdata, event.ydata)
                self.plot_view.draw()

    def on_release(self, event):
        self.button_position = None
        self.artist = None

    def on_pick(self, event):
        if isinstance(event.artist, matplotlib.lines.Line2D) and event.mouseevent.button == 1:
            self.plot_view.toggleLine(event.artist)
        elif isinstance(event.artist, matplotlib.text.Annotation) and event.mouseevent.button == 1:
            self.artist = event.artist
            self.button_position = (event.mouseevent.x, event.mouseevent.y)
            return True
        elif isinstance(event.artist, matplotlib.text.Annotation) and event.mouseevent.button == 3:
            self.artist = event.artist
            self.plot_view.removeAnnotation(self.artist)
            self.plot_view.draw()

    def motion_notify_event(self, event):
        if self.artist is None:
            self.plot_view.displayToolTip(event)
        elif isinstance(self.artist, matplotlib.text.Annotation):
            if not event.xdata is None and not event.ydata is None:
                self.artist.xytext = (event.xdata, event.ydata)
                self.plot_view.draw()


