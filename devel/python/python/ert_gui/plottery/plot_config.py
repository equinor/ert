import itertools


class PlotConfig(object):

    def __init__(self, title="Unnamed", x_label=None, y_label=None):
        super(PlotConfig, self).__init__()
        self.__title = title

        self.setLineColorCycle(["#386CB0", "#7FC97F", "#FDC086", "#F0027F", "#BF5B17"])

        self.__legend_items = []
        self.__legend_labels = []

        self.__x_label = x_label
        self.__y_label = y_label

        self.__line_color = None
        self.__line_style = "-"
        self.__line_alpha = 0.8
        self.__line_marker = None


    def lineColor(self):
        if self.__line_color is None:
            self.nextColor()

        return self.__line_color

    def nextColor(self):
        self.__line_color = self.__line_color_cycle.next()
        return self.lineColor()

    def setLineColorCycle(self, color_list):
        self.__line_color_cycle = itertools.cycle(color_list)

    def addLegendItem(self, label, item):
        self.__legend_items.append(item)
        self.__legend_labels.append(label)

    def title(self):
        """ :rtype: str """
        return self.__title

    def lineStyle(self):
        return self.__line_style

    def lineAlpha(self):
        return self.__line_alpha

    def lineMarker(self):
        return self.__line_marker

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