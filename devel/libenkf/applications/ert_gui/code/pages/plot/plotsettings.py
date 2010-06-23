from pages.plot.plotconfig import PlotConfig
import matplotlib
from erttypes import time_t
import datetime

class PlotSettings:
    plot_color = (55/255.0, 126/255.0, 200/255.0) # bluish
    selected_color = (152/255.0, 78/255.0, 163/255.0) # purple
    history_color = (255/255.0, 0/255.0, 0/255.0) # red
    refcase_color = (0/255.0, 200/255.0, 0/255.0) # green

    def __init__(self):
        self.xminf = 0.0
        self.xmaxf = 1.0
        self.x_limits = (None, None)

        self.yminf = 0.0
        self.ymaxf = 1.0
        self.y_limits = (None, None)

        self.plot_path = "."
        self.plot_config_path = "."

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

        self._plot_config_dict = {}
        for pc in self._plot_configs:
            self._plot_config_dict[pc.name] = pc

        self.selected_members = []

    def getPlotConfigList(self):
        return self._plot_configs

    def getPlotConfigDict(self):
        return self._plot_config_dict

    def getLimitsTuple(self):
        return (self.x_limits[0], self.x_limits[1], self.y_limits[0], self.y_limits[1])

    def getZoomTuple(self):
        return (self.xminf, self.xmaxf, self.yminf, self.ymaxf)

    def selectMember(self, member):
        self.selected_members.append(int(member))

    def unselectMember(self, member):
        member = int(member)
        if member in self.selected_members:
            self.selected_members.remove(member)

        print self.selected_members

    def clearMemberSelection(self):
        self.selected_members = []

    def getSelectedMembers(self):
        return self.selected_members

    def __convertDate(self, ert_time):
        if ert_time is None:
            ert_time = time_t(0)

        if isinstance(ert_time, datetime.date):
            return matplotlib.dates.date2num(ert_time)
        else:
            return matplotlib.dates.date2num(ert_time.datetime())

    def setMinYLimit(self, value):
        self.y_limits = (value, self.y_limits[1])

    def getMinYLimit(self, y_min, type=""):
        if self.y_limits[0] is None:
            return y_min
        else:
            return self.y_limits[0]

    def setMaxYLimit(self, value):
        self.y_limits = (self.y_limits[0], value)

    def getMaxYLimit(self, y_max, type=""):
        if self.y_limits[1] is None:
            return y_max
        else:
            return self.y_limits[1]

    def setMinXLimit(self, value):
        self.x_limits = (value, self.x_limits[1])

    def getMinXLimit(self, x_min, type):
        """Returns the provided x_min value if the custom x_min value is None. Converts dates to numbers"""
        if self.x_limits[0] is None:
            x_limit = x_min
        else:
            x_limit = self.x_limits[0]

            if type == "time":
                x_limit = time_t(long(x_limit * 86400))

        if type == "time":
            x_limit = self.__convertDate(x_limit)

        return x_limit

    def setMaxXLimit(self, value):
        self.x_limits = (self.x_limits[0], value)

    def getMaxXLimit(self, x_max, type):
        if self.x_limits[1] is None:
            x_limit = x_max
        else:
            x_limit = self.x_limits[1]
            if type == "time":
                x_limit = time_t(long(x_limit * 86400))

        if type == "time":
            x_limit = self.__convertDate(x_limit)

        return x_limit

    def getLimitStates(self):
        x_min_state = not self.x_limits[0] is None
        x_max_state = not self.x_limits[1] is None
        y_min_state = not self.y_limits[0] is None
        y_max_state = not self.y_limits[1] is None
        return (x_min_state, x_max_state, y_min_state, y_max_state)




