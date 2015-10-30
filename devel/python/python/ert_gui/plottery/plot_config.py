import itertools
from ert_gui.plottery import PlotStyle


class PlotConfig(object):

    def __init__(self, title="Unnamed", x_label=None, y_label=None):
        super(PlotConfig, self).__init__()
        self.__title = title

        self.__line_color_cycle = itertools.cycle(["#000000"]) #Black
        # Blueish, Greenlike, Beigeoid, Pinkness, Orangy-Brown
        self.setLineColorCycle(["#386CB0", "#7FC97F", "#FDC086", "#F0027F", "#BF5B17"])

        self.__legend_items = []
        self.__legend_labels = []

        self.__x_label = x_label
        self.__y_label = y_label

        self.__default_style = PlotStyle(name="Default", color=None, alpha=0.8)
        self.__refcase_style = PlotStyle(name="Refcase", alpha=0.8, marker="x", width=2.0)
        self.__observation_style = PlotStyle(name="Observations")
        self.__histogram_style = PlotStyle(name="Histogram", width=2.0)
        self.__distribution_style = PlotStyle(name="Distribution", line_style="", marker="o", alpha=0.8, width=10.0)
        self.__current_color = None

        self.__legend_enabled = True
        self.__grid_enabled = True
        self.__date_support_active = True

        self.__statistics_style = {
            "mean": PlotStyle("Mean", line_style="", alpha=0.8),
            "p50": PlotStyle("P50", line_style="", alpha=0.8),
            "min-max": PlotStyle("Min/Max", line_style="", alpha=0.8),
            "p10-p90": PlotStyle("P10-P90", line_style="", alpha=0.8),
            "p33-p67": PlotStyle("P33-P67", line_style="", alpha=0.8)
        }

    def currentColor(self):
        if self.__current_color is None:
            self.nextColor()

        return self.__current_color

    def nextColor(self):
        self.__current_color = self.__line_color_cycle.next()
        return self.__current_color

    def setLineColorCycle(self, color_list):
        self.__line_color_cycle = itertools.cycle(color_list)

    def addLegendItem(self, label, item):
        self.__legend_items.append(item)
        self.__legend_labels.append(label)

    def title(self):
        """ :rtype: str """
        return self.__title

    def defaultStyle(self):
        style = PlotStyle("Default Style")
        style.copyStyleFrom(self.__default_style)
        style.color = self.currentColor()
        return style

    def observationsStyle(self):
        """ @rtype: PlotStyle """
        style = PlotStyle("Observations Style")
        style.copyStyleFrom(self.__observation_style)
        return style

    def refcaseStyle(self):
        """ @rtype: PlotStyle """
        style = PlotStyle("Refcase Style")
        style.copyStyleFrom(self.__refcase_style)
        return style

    def histogramStyle(self):
        """ @rtype: PlotStyle """
        style = PlotStyle("Histogram Style")
        style.copyStyleFrom(self.__histogram_style)
        style.color = self.currentColor()
        return style

    def distributionStyle(self):
        """ @rtype: PlotStyle """
        style = PlotStyle("Distribution Style")
        style.copyStyleFrom(self.__distribution_style)
        style.color = self.currentColor()
        return style

    def xLabel(self):
        return self.__x_label

    def yLabel(self):
        return self.__y_label

    def legendItems(self):
        return self.__legend_items

    def legendLabels(self):
        return self.__legend_labels

    def setXLabel(self, label):
        self.__x_label = label

    def setYLabel(self, label):
        self.__y_label = label

    def setObservationsEnabled(self, enabled):
        self.__observation_style.setEnabled(enabled)

    def isObservationsEnabled(self):
        return self.__observation_style.isEnabled()

    def setRefcaseEnabled(self, enabled):
        self.__refcase_style.setEnabled(enabled)

    def isRefcaseEnabled(self):
        return self.__refcase_style.isEnabled()

    def isLegendEnabled(self):
        return self.__legend_enabled

    def setLegendEnabled(self, enabled):
        self.__legend_enabled = enabled

    def isGridEnabled(self):
        return self.__grid_enabled

    def setGridEnabled(self, enabled):
        self.__grid_enabled = enabled

    def deactivateDateSupport(self):
        self.__date_support_active = False

    def isDateSupportActive(self):
        return self.__date_support_active

    def setStatisticsStyle(self, statistic, line_style, marker):
        style = self.__statistics_style[statistic]
        style.line_style = line_style
        style.marker = marker

    def getStatisticsStyle(self, statistic):
        style = self.__statistics_style[statistic]
        copy_style = PlotStyle(style.name)
        copy_style.copyStyleFrom(style)
        copy_style.color = self.currentColor()
        return copy_style

    def setRefcaseStyle(self, line_style, marker):
        self.__refcase_style.line_style = line_style
        self.__refcase_style.marker = marker

    def setDefaultStyle(self, line_style, marker):
        self.__default_style.line_style = line_style
        self.__default_style.marker = marker