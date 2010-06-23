from matplotlib.figure import Figure
import matplotlib.lines
import matplotlib.text
from matplotlib.dates import AutoDateLocator
from plotter import Plotter
import numpy

class PlotFigure:
    def __init__(self):
        self.fig = Figure(dpi=100)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlim()
        self.plotter = Plotter()
        self.annotations = []

    def getFigure(self):
        return self.fig

    def drawPlot(self, data, plot_settings):
        self.axes.cla()
        self.lines = []

        self.axes.set_title(data.getTitle())

        if data.hasInvertedYAxis() and not self.axes.yaxis_inverted():
            self.axes.invert_yaxis()
        elif not data.hasInvertedYAxis() and self.axes.yaxis_inverted():
            self.axes.invert_yaxis()

        old_annotations = self.annotations
        self.annotations = []
        for annotation in old_annotations:
            x, y = annotation.xy
            xt, yt = annotation.xytext
            self.annotate(annotation.get_text(), x, y, xt, yt)

        selected_members = plot_settings.getSelectedMembers()

        for member in data.x_data.keys():
            x, y, x_std, y_std = self.__setupData(data, data.x_data[member], data.y_data[member])

            if member in selected_members:
                plot_config = plot_settings.selected_plot_config
            else:
                plot_config = plot_settings.plot_config

            if data.getXDataType() == "time":
                line = self.plotter.plot_date(self.axes, plot_config, x, y)
            else:
                line = self.plotter.plot(self.axes, plot_config, x, y)

            line.set_gid(member)
            self.lines.append(line)

        if not data.obs_x is None and not data.obs_y is None:
            x, y, x_std, y_std = self.__setupData(data, data.obs_x, data.obs_y, data.obs_std_x, data.obs_std_y)

            if data.getXDataType() == "time":
                self.plotter.plot_date(self.axes, plot_settings.observation_plot_config, x, y)
            else:
                self.plotter.plot(self.axes, plot_settings.observation_plot_config, x, y)

            if not data.obs_std_x is None or not data.obs_std_y is None:
                if plot_settings.std_plot_config.is_visible:
                    if data.getXDataType() == "time":
                        if not y_std is None:
                            self.plotter.plot_date(self.axes, plot_settings.std_plot_config, x, y - y_std)
                            self.plotter.plot_date(self.axes, plot_settings.std_plot_config, x, y + y_std)
                        elif not x_std is None:
                            self.plotter.plot_date(self.axes, plot_settings.std_plot_config, x - x_std, y)
                            self.plotter.plot_date(self.axes, plot_settings.std_plot_config, x + x_std, y)
                    else:
                        if not y_std is None:
                            self.plotter.plot(self.axes, plot_settings.std_plot_config, x, y - y_std)
                            self.plotter.plot(self.axes, plot_settings.std_plot_config, x, y + y_std)
                        elif not x_std is None:
                            self.plotter.plot(self.axes, plot_settings.std_plot_config, x - x_std, y)
                            self.plotter.plot(self.axes, plot_settings.std_plot_config, x + x_std, y)

                if  plot_settings.errorbar_plot_config.is_visible:
                    self.plotter.plot_errorbar(self.axes, plot_settings.errorbar_plot_config, x, y, x_std, y_std)

        if not data.refcase_x is None and not data.refcase_y is None and plot_settings.refcase_plot_config.is_visible:
            x, y, x_std, y_std = self.__setupData(data, data.refcase_x, data.refcase_y)

            if data.getXDataType() == "time":
                self.plotter.plot_date(self.axes, plot_settings.refcase_plot_config, x, y)

        if data.getXDataType() == "time":
            yearsFmt = matplotlib.dates.DateFormatter('%b \'%Y')
            self.axes.xaxis.set_major_formatter(yearsFmt)
            self.fig.autofmt_xdate()

        number_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        number_formatter.set_scientific(True)
        self.axes.yaxis.set_major_formatter(number_formatter)

        self.updateLimits(plot_settings, data)

    def __setupData(self, data, x, y, std_x = None, std_y = None):
        if data.getXDataType() == "time":
            x = [t.datetime() for t in x]

        if not std_x is None:
            std_x = numpy.array(std_x)

        if not std_y is None:
            std_y = numpy.array(std_y)

        x = numpy.array(x)
        y = numpy.array(y)

        return x, y, std_x, std_y

    def updateLimits(self, plot_settings, data):
        self.setXViewFactors(plot_settings, plot_settings.xminf, plot_settings.xmaxf, data.x_min, data.x_max, data.getXDataType())
        self.setYViewFactors(plot_settings, plot_settings.yminf, plot_settings.ymaxf, data.y_min, data.y_max, data.getYDataType())
        
    def annotate(self, label, x, y, xt=None, yt=None):
        coord = (x, y)
        xytext = None
        if not xt is None and not yt is None:
            xytext = (xt, yt)
        #arrow = dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        arrow = dict(arrowstyle="->")
        annotation = self.axes.annotate(str(label), coord, xytext=xytext, xycoords='data', textcoords='data', arrowprops=arrow, picker=1)
        self.annotations.append(annotation)

    def removeAnnotation(self, annotation):
        self.axes.texts.remove(annotation)
        self.annotations.remove(annotation)

    def clearAnnotations(self):
        for annotation in self.annotations:
            self.removeAnnotation(annotation)

    def setXLimits(self, x_min, x_max):
        self.axes.set_xlim(x_min, x_max)

    def setYLimits(self, y_min, y_max):
        self.axes.set_ylim(y_min, y_max)

    def setXViewFactors(self, plot_settings, xminf, xmaxf, x_min, x_max, type):
        plot_settings.xminf = xminf
        plot_settings.xmaxf = xmaxf

        x_min = plot_settings.getMinXLimit(x_min, type)
        x_max = plot_settings.getMaxXLimit(x_max, type)

        if not x_min is None and not x_max is None:
            range = x_max - x_min
            self.setXLimits(x_min + xminf * range - range*0.05, x_min + xmaxf * range + range*0.05)

    def setYViewFactors(self, plot_settings, yminf, ymaxf, y_min, y_max, type=""):
        plot_settings.yminf = yminf
        plot_settings.ymaxf = ymaxf

        y_min = plot_settings.getMinYLimit(y_min, type)
        y_max = plot_settings.getMaxYLimit(y_max, type)

        if not y_min is None and not y_max is None:
            range = y_max - y_min
            self.setYLimits(y_min + yminf * range - range*0.05, y_min + ymaxf * range + range*0.05)

    def getAnnotations(self):
        """Creates a list of tuples describing all annotations. (label, x, y, x_text, y_text)"""
        result = []
        for annotation in self.annotations:
            label = annotation.get_text()
            x, y = annotation.xy
            xt, yt = annotation.xytext

            result.append((label, x, y, xt, yt))
        return result





