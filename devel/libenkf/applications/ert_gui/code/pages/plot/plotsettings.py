from pages.plot.plotconfig import PlotConfig
import matplotlib
from erttypes import time_t
import datetime
from  PyQt4.QtCore import QObject, SIGNAL

class PlotSettings(QObject):
    plot_color = (55/255.0, 126/255.0, 200/255.0) # bluish
    selected_color = (152/255.0, 78/255.0, 163/255.0) # purple
    history_color = (255/255.0, 0/255.0, 0/255.0) # red
    refcase_color = (0/255.0, 200/255.0, 0/255.0) # green

    def __init__(self):
        QObject.__init__(self)

        self._xminf = 0.0
        self._xmaxf = 1.0
        self._x_limits = (None, None)

        self._yminf = 0.0
        self._ymaxf = 1.0
        self._y_limits = (None, None)

        self._plot_path = "."
        self._plot_config_path = "."

        self.observation_plot_config = PlotConfig("Observation", color = self.history_color, zorder=10)
        self.refcase_plot_config = PlotConfig("Refcase", visible=False, color = self.refcase_color, zorder=10)
        self.std_plot_config = PlotConfig("Error", linestyle=":", visible=False, color = self.history_color, zorder=10)
        self.plot_config = PlotConfig("Members", color = self.plot_color, alpha=0.125, zorder=1, picker=2)
        self.selected_plot_config = PlotConfig("Selected members", color = self.selected_color, alpha=0.5, zorder=8,
                                               picker=2)
        self.errorbar_plot_config = PlotConfig("Errorbars", visible=False, color = self.history_color, alpha=0.5,
                                               zorder=10)

        self._plot_configs = [self.plot_config,
                              self.selected_plot_config,
                              self.refcase_plot_config,
                              self.observation_plot_config,
                              self.std_plot_config,
                              self.errorbar_plot_config]

        for pc in self._plot_configs:
            self.connect(pc.signal_handler, SIGNAL('plotConfigChanged(PlotConfig)'), self.notify)

        self._plot_config_dict = {}
        for pc in self._plot_configs:
            self._plot_config_dict[pc.name] = pc

        self._selected_members = []

        self._annotations = []

    def notify(self, *args):
        self.emit(SIGNAL('plotSettingsChanged(PlotSettings)'), self)

    def getPlotConfigList(self):
        return self._plot_configs

    def getPlotConfigDict(self):
        return self._plot_config_dict

    def getLimitsTuple(self):
        return (self._x_limits[0], self._x_limits[1], self._y_limits[0], self._y_limits[1])

    def getZoomTuple(self):
        return (self._xminf, self._xmaxf, self._yminf, self._ymaxf)

    def selectMember(self, member):
        if not member in self._selected_members:
            self._selected_members.append(int(member))
            self.notify()

    def unselectMember(self, member):
        member = int(member)
        if member in self._selected_members:
            self._selected_members.remove(member)
            self.notify()

    def clearMemberSelection(self):
        self._selected_members = []
        self.notify()

    def getSelectedMembers(self):
        return self._selected_members

    def setMinYLimit(self, value):
        if not value == self._y_limits[0]:
            self._y_limits = (value, self._y_limits[1])
            self.notify()

    def getMinYLimit(self, y_min, data_type=""):
        if self._y_limits[0] is None:
            return y_min
        else:
            return self._y_limits[0]

    def setMaxYLimit(self, value):
        if not value == self._y_limits[1]:
            self._y_limits = (self._y_limits[0], value)
            self.notify()

    def getMaxYLimit(self, y_max, data_type=""):
        if self._y_limits[1] is None:
            return y_max
        else:
            return self._y_limits[1]

    def setMinXLimit(self, value):
        if not value == self._x_limits[0]:
            self._x_limits = (value, self._x_limits[1])
            self.notify()

    def getMinXLimit(self, x_min, data_type):
        """Returns the provided x_min value if the custom x_min value is None. Converts dates to numbers"""
        if self._x_limits[0] is None:
            x_limit = x_min
        else:
            x_limit = self._x_limits[0]

        if not x_limit is None and data_type == "time" and not isinstance(x_limit, time_t):
            x_limit = time_t(long(round(x_limit)))

        return x_limit

    def setMaxXLimit(self, value):
        if not value == self._x_limits[1]:
            self._x_limits = (self._x_limits[0], value)
            self.notify()

    def getMaxXLimit(self, x_max, data_type):
        if self._x_limits[1] is None:
            x_limit = x_max
        else:
            x_limit = self._x_limits[1]

        if not x_limit is None and data_type == "time" and not isinstance(x_limit, time_t):
            x_limit = time_t(long(round(x_limit)))

        return x_limit

    def getLimitStates(self):
        x_min_state = not self._x_limits[0] is None
        x_max_state = not self._x_limits[1] is None
        y_min_state = not self._y_limits[0] is None
        y_max_state = not self._y_limits[1] is None
        return (x_min_state, x_max_state, y_min_state, y_max_state)

    def setPlotPath(self, plot_path):
        if not plot_path == self._plot_path:
            self._plot_path = plot_path
            self.notify()

    def setPlotConfigPath(self, plot_config_path):
        if not plot_config_path == self._plot_config_path:
            self._plot_config_path = plot_config_path
            self.notify()

    def getPlotPath(self):
        return self._plot_path

    def getPlotConfigPath(self):
        return self._plot_config_path

    def setMinXZoom(self, value):
        if not self._xminf == value:
            self._xminf = value
            self.notify()

    def setMaxXZoom(self, value):
        if not self._xmaxf == value:
            self._xmaxf = value
            self.notify()

    def setMinYZoom(self, value):
        if not self._yminf == value:
            self._yminf = value
            self.notify()

    def setMaxYZoom(self, value):
        if not self._ymaxf == value:
            self._ymaxf = value
            self.notify()

    def getMinXZoom(self):
        return self._xminf

    def getMaxXZoom(self):
        return self._xmaxf

    def getMinYZoom(self):
        return self._yminf

    def getMaxYZoom(self):
        return self._ymaxf

    def getAnnotations(self):
        return self._annotations

    def clearAnnotations(self):
        if len(self._annotations) > 0:
            self._annotations = []
            self.notify()

    def addAnnotation(self, label, x, y, xt, yt):
        annotation = PlotAnnotation(label, x, y, xt, yt)
        self._annotations.append(annotation)
        self.notify()
        return annotation

    def removeAnnotation(self, annotation):
        if annotation in self._annotations:
            self._annotations.remove(annotation)
            self.notify()


class PlotAnnotation:
    def __init__(self, label, x, y, xt, yt):
        self.label = label
        self.x = x
        self.y = y
        self.xt = xt
        self.yt = yt

    def setUserData(self, user_data):
        self._user_data = user_data

    def getUserData(self):
        return self._user_data








