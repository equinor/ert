import os

from PyQt4.QtCore import QUrl, Qt, pyqtSignal
from PyQt4.QtGui import QWidget, QGridLayout, QPainter
from PyQt4.QtWebKit import QWebView, QWebSettings

from ert_gui.tools.plot import PlotBridge
from ert_gui.tools.plot.plot_bridge import PlotWebPage


class PlotWebView(QWebView):
    def __init__(self, name):
        QWebView.__init__(self)
        self.setPage(PlotWebPage(name))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.settings().setAttribute(QWebSettings.JavascriptEnabled, True)
        self.settings().setAttribute(QWebSettings.LocalContentCanAccessFileUrls, True)
        self.settings().setAttribute(QWebSettings.LocalContentCanAccessRemoteUrls, True)
        self.settings().clearMemoryCaches()



class PlotPanel(QWidget):
    plotReady = pyqtSignal()



    def __init__(self, name, debug_name, plot_url):
        QWidget.__init__(self)

        self.__name = name
        self.__debug_name = debug_name
        self.__plot_url = plot_url

        layout = QGridLayout()

        self.web_view = PlotWebView(debug_name)

        layout.addWidget(self.web_view)
        self.setLayout(layout)

        self.__plot_is_visible = True
        self.__plot_bridge = PlotBridge(self.getWebView().page(), plot_url)
        self.__plot_bridge.plotReady.connect(self.plotReady)


    def getName(self):
        return self.__name

    def getUrl(self):
        return self.__plot_url

    def getWebView(self):
        return self.web_view

    def setSettings(self, settings):
        if self.isPlotVisible():
            self.__plot_bridge.setPlotSettings(settings)

    def isReady(self):
        return self.__plot_bridge.isReady()


    def resizeEvent(self, event):
        QWidget.resizeEvent(self, event)
        if self.isPlotVisible():
            self.__plot_bridge.updatePlotSize(size = self.size())


    def supportsPlotProperties(self, time=False, value=False, depth=False, histogram=False):
        return self.__plot_bridge.supportsPlotProperties(time, value, depth, histogram)

    def isPlotVisible(self):
        return self.__plot_is_visible

    def setPlotIsVisible(self, visible):
        self.__plot_is_visible = visible

    def getPlotBridge(self):
        """ @rtype: PlotBridge """
        return self.__plot_bridge

    def renderNow(self):
        if self.isPlotVisible():
            self.__plot_bridge.renderNow()
