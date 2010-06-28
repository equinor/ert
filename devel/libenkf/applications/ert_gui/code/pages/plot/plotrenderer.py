from pages.plot.plotter import Plotter
from matplotlib.dates import AutoDateLocator, datetime, matplotlib
import erttypes

class PlotRenderer:

    def __init__(self):
        self.plotter = Plotter()

    def drawPlot(self, axes, data, plot_settings):
        axes.set_title(data.getTitle())

        if data.hasInvertedYAxis() and not axes.yaxis_inverted():
            axes.invert_yaxis()
        elif not data.hasInvertedYAxis() and axes.yaxis_inverted():
            axes.invert_yaxis()


        annotations = plot_settings.getAnnotations()
        for annotation in annotations:
            coord = (annotation.x, annotation.y)
            xytext = None
            if not annotation.xt is None and not annotation.yt is None:
                xytext = (annotation.xt, annotation.yt)
            arrow = dict(arrowstyle="->")
            label = str(annotation.label)
            annotation_artist = axes.annotate(label, coord, xytext=xytext, xycoords='data', textcoords='data', arrowprops=arrow, picker=1)
            annotation.setUserData(annotation_artist)

        self._plotMembers(axes, data, plot_settings)

        if not data.obs_x is None and not data.obs_y is None:
            self._plotObservations(axes, data, plot_settings)

        if not data.obs_std_x is None or not data.obs_std_y is None:
            self._plotError(axes, data, plot_settings)

        if not data.refcase_x is None and not data.refcase_y is None and plot_settings.refcase_plot_config.is_visible:
            self._plotRefCase(axes, data, plot_settings)
            
        if data.getXDataType() == "time":
            yearsFmt = matplotlib.dates.DateFormatter('%b \'%Y')
            axes.xaxis.set_major_formatter(yearsFmt)
#            labels = axes.get_xticklabels()
#            for label in labels:
#                label.set_rotation(30)


        number_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        number_formatter.set_scientific(True)
        axes.yaxis.set_major_formatter(number_formatter)

        self._updateLimitsAndZoom(axes, plot_settings, data)

    def _plotMembers(self, axes, data, plot_settings):
        selected_members = plot_settings.getSelectedMembers()
        for member in data.x_data.keys():
            x = data.x_data[member]
            y = data.y_data[member]
            if member in selected_members:
                plot_config = plot_settings.selected_plot_config
            else:
                plot_config = plot_settings.plot_config
            if data.getXDataType() == "time":
                line = self.plotter.plot_date(axes, plot_config, x, y)
            else:
                line = self.plotter.plot(axes, plot_config, x, y)
            line.set_gid(member)

    def _plotError(self, axes, data, plot_settings):
        x = data.obs_x
        y = data.obs_y
        x_std = data.obs_std_x
        y_std = data.obs_std_y

        if plot_settings.std_plot_config.is_visible:
            if data.getXDataType() == "time":
                if not y_std is None:
                    self.plotter.plot_date(axes, plot_settings.std_plot_config, x, y - y_std)
                    self.plotter.plot_date(axes, plot_settings.std_plot_config, x, y + y_std)
                elif not x_std is None:
                    self.plotter.plot_date(axes, plot_settings.std_plot_config, x - x_std, y)
                    self.plotter.plot_date(axes, plot_settings.std_plot_config, x + x_std, y)
            else:
                if not y_std is None:
                    self.plotter.plot(axes, plot_settings.std_plot_config, x, y - y_std)
                    self.plotter.plot(axes, plot_settings.std_plot_config, x, y + y_std)
                elif not x_std is None:
                    self.plotter.plot(axes, plot_settings.std_plot_config, x - x_std, y)
                    self.plotter.plot(axes, plot_settings.std_plot_config, x + x_std, y)

        if  plot_settings.errorbar_plot_config.is_visible:
            self.plotter.plot_errorbar(axes, plot_settings.errorbar_plot_config, x, y, x_std, y_std)

    def _plotObservations(self, axes, data, plot_settings):
        x = data.obs_x
        y = data.obs_y

        if data.getXDataType() == "time":
            self.plotter.plot_date(axes, plot_settings.observation_plot_config, x, y)
        else:
            self.plotter.plot(axes, plot_settings.observation_plot_config, x, y)

    def _plotRefCase(self, axes, data, plot_settings):
        x = data.refcase_x
        y = data.refcase_y
        if data.getXDataType() == "time":
            self.plotter.plot_date(axes, plot_settings.refcase_plot_config, x, y)

    def _updateLimitsAndZoom(self, axes, plot_settings, data):
        self._setXViewFactors(axes, plot_settings, data)
        self._setYViewFactors(axes, plot_settings, data)

    def __convertDate(self, ert_time):
        if ert_time is None:
            ert_time = erttypes.time_t(0)

        if isinstance(ert_time, datetime.date):
            return matplotlib.dates.date2num(ert_time)
        else:
            return matplotlib.dates.date2num(ert_time.datetime())

    def _setXViewFactors(self, axes, plot_settings, data):
        xminf = plot_settings.getMinXZoom()
        xmaxf = plot_settings.getMaxXZoom()

        x_min = plot_settings.getMinXLimit(data.x_min, data.getXDataType())
        x_max = plot_settings.getMaxXLimit(data.x_max, data.getXDataType())

        if data.getXDataType() == "time":
            x_min = self.__convertDate(x_min)
            x_max = self.__convertDate(x_max)

        if not x_min is None and not x_max is None:
            range = x_max - x_min
            axes.set_xlim(x_min + xminf * range - range*0.0, x_min + xmaxf * range + range*0.0)

    def _setYViewFactors(self, axes, plot_settings, data):
        yminf = plot_settings.getMinYZoom()
        ymaxf = plot_settings.getMaxYZoom()

        y_min = plot_settings.getMinYLimit(data.y_min, data.getYDataType())
        y_max = plot_settings.getMaxYLimit(data.y_max, data.getYDataType())


        if not y_min is None and not y_max is None:
            range = y_max - y_min
            axes.set_ylim(y_min + yminf * range - range*0.0, y_min + ymaxf * range + range*0.0)