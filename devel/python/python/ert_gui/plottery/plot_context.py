from .plot_config import PlotConfig

class PlotContext(object):

    def __init__(self, ert, figure, plot_config, cases, key):
        super(PlotContext, self).__init__()
        self.__key = key
        self.__cases = cases
        self.__figure = figure
        self.__ert = ert
        self.__plot_config = plot_config

    def figure(self):
        """ :rtype: matplotlib.figure.Figure"""
        return self.__figure

    def plotConfig(self):
        """ :rtype: PlotConfig """
        return self.__plot_config

    def ert(self):
        """ :rtype: ert.enkf.EnKFMain"""
        return self.__ert

    def cases(self):
        """ :rtype: list of str """
        return self.__cases

    def key(self):
        """ :rtype: str """
        return self.__key


