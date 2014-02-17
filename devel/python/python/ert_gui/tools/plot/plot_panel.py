import json
import os
from PyQt4.QtCore import QUrl, Qt, pyqtSlot, pyqtSignal, QObject
from PyQt4.QtGui import QWidget, QGridLayout, QPainter
from PyQt4.QtWebKit import QWebView, QWebPage, QWebSettings
from ert_gui.tools.plot.data import PlotData


class PlotWebPage(QWebPage):
    def __init__(self, name):
        QWebPage.__init__(self)
        self.name = name

    def javaScriptConsoleMessage(self, message, line_number, source_id):
        print("[%s] Source: %s at line: %d -> %s" % (self.name, source_id, line_number, message))


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
        self.__ready = False
        self.__html_ready = False
        self.__data = PlotData("invalid", parent=self)

        root_path = os.getenv("ERT_SHARE_PATH")
        path = os.path.join(root_path, plot_url)

        layout = QGridLayout()

        self.web_view = PlotWebView(debug_name)
        self.applyContextObject()
        # self.web_view.page().mainFrame().javaScriptWindowObjectCleared.connect(self.applyContextObject)
        self.web_view.loadFinished.connect(self.loadFinished)

        layout.addWidget(self.web_view)

        self.setLayout(layout)

        self.web_view.setUrl(QUrl("file://%s" % path))

        self.__plot_is_visible = True


    @pyqtSlot(result=QObject)
    def getPlotData(self):
        return self.__data

    @pyqtSlot()
    def htmlInitialized(self):
        # print("[%s] Initialized!" % self.__name)
        self.__html_ready = True
        self.checkStatus()

    def setPlotData(self, data):
        if self.isPlotVisible():
            self.__data = data
            self.web_view.page().mainFrame().evaluateJavaScript("updatePlot();")

    def setScales(self, time_min, time_max, value_min, value_max, depth_min, depth_max):
        if self.isPlotVisible():
            if value_min is None:
                value_min = "null"

            if value_max is None:
                value_max = "null"

            if time_min is None:
                time_min = "null"
            else:
                time_min = time_min.ctime()

            if time_max is None:
                time_max = "null"
            else:
                time_max = time_max.ctime()

            if depth_min is None:
                depth_min = "null"

            if depth_max is None:
                depth_max = "null"

            scales = (time_min, time_max, value_min, value_max, depth_min, depth_max)
            self.web_view.page().mainFrame().evaluateJavaScript("setScales(%s,%s,%s,%s,%s,%s);" % scales)

    def setReportStepTime(self, report_step_time):
        if self.isPlotVisible():
            if report_step_time is None:
                report_step_time = "null"

            self.web_view.page().mainFrame().evaluateJavaScript("setReportStepTime(%s);" % (report_step_time))


    def applyContextObject(self):
        self.web_view.page().mainFrame().addToJavaScriptWindowObject("plot_data_source", self)


    def loadFinished(self, ok):
        self.__ready = True
        self.checkStatus()

    def checkStatus(self):
        if self.__ready and self.__html_ready:
            # print("[%s] Ready!" % self.__name)
            self.plotReady.emit()
            self.updatePlotSize()


    def isReady(self):
        return self.__ready and self.__html_ready


    def resizeEvent(self, event):
        QWidget.resizeEvent(self, event)

        if self.isReady():
            self.updatePlotSize()


    def updatePlotSize(self):
        if self.isPlotVisible():
            size = self.size()
            self.web_view.page().mainFrame().evaluateJavaScript("setSize(%d,%d);" % (size.width(), size.height()))

    def supportsPlotProperties(self, time=False, value=False, depth=False, histogram=False):
        time = str(time).lower()
        value = str(value).lower()
        depth = str(depth).lower()
        histogram = str(histogram).lower()
        return self.web_view.page().mainFrame().evaluateJavaScript("supportsPlotProperties(%s,%s,%s,%s);" % (time, value, depth, histogram)).toBool()

    def getName(self):
        return self.__name

    def isPlotVisible(self):
        return self.__plot_is_visible

    def setPlotIsVisible(self, visible):
        self.__plot_is_visible = visible

    def setCustomSettings(self, settings):
        json_settings = json.dumps(settings)
        self.web_view.page().mainFrame().evaluateJavaScript("setCustomSettings(%s);" % json_settings)


