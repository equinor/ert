import json
import os
from PyQt4.QtCore import QObject, pyqtSlot, pyqtSignal, QUrl
from PyQt4.QtWebKit import QWebPage
from ert_gui.tools.plot.data import PlotData

class PlotWebPage(QWebPage):
    def __init__(self, name):
        QWebPage.__init__(self)
        self.name = name

    def javaScriptConsoleMessage(self, message, line_number, source_id):
        print("[%s] Source: %s at line: %d -> %s" % (self.name, source_id, line_number, message))


class PlotBridge(QObject):
    plotReady = pyqtSignal()
    renderingFinished = pyqtSignal()

    def __init__(self, web_page, plot_url):
        QObject.__init__(self)
        assert isinstance(web_page, QWebPage)

        self.__web_page = web_page
        self.__ready = False
        self.__html_ready = False
        self.__data = PlotData("invalid", parent=self)
        self.__size = None

        self.applyContextObject()

        root_path = os.getenv("ERT_SHARE_PATH")
        path = os.path.join(root_path, plot_url)
        self.__web_page.mainFrame().load(QUrl("file://%s" % path))
        self.__web_page.loadFinished.connect(self.loadFinished)


    def applyContextObject(self):
        self.__web_page.mainFrame().addToJavaScriptWindowObject("plot_data_source", self)

    def updatePlotSize(self, size):
        self.__size = size
        if self.isReady():
            self.__web_page.mainFrame().evaluateJavaScript("setSize(%d,%d);" % (size.width(), size.height()))
            self.renderNow()

    def supportsPlotProperties(self, time=False, value=False, depth=False, histogram=False):
        time = str(time).lower()
        value = str(value).lower()
        depth = str(depth).lower()
        histogram = str(histogram).lower()
        return self.__web_page.mainFrame().evaluateJavaScript("supportsPlotProperties(%s,%s,%s,%s);" % (time, value, depth, histogram)).toBool()

    def setPlotData(self, data):
        self.__data = data
        self.__web_page.mainFrame().evaluateJavaScript("updatePlot();")

    def setReportStepTime(self, report_step_time):
        if report_step_time is None:
            report_step_time = "null"

        self.__web_page.mainFrame().evaluateJavaScript("setReportStepTime(%s);" % (report_step_time))


    def setScales(self, time_min, time_max, value_min, value_max, depth_min, depth_max):
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
        self.__web_page.mainFrame().evaluateJavaScript("setScales(%s,%s,%s,%s,%s,%s);" % scales)

    @pyqtSlot(result=QObject)
    def getPlotData(self):
        return self.__data

    @pyqtSlot()
    def htmlInitialized(self):
        # print("[%s] Initialized!" % self.__name)
        self.__html_ready = True
        self.checkStatus()

    def loadFinished(self, ok):
        self.__ready = True
        self.checkStatus()

    def checkStatus(self):
        if self.__ready and self.__html_ready:
            # print("[%s] Ready!" % self.__name)
            self.plotReady.emit()
            if self.__size is not None:
                self.updatePlotSize(self.__size)

    def isReady(self):
        return self.__ready and self.__html_ready

    def getPrintWidth(self):
        return self.__web_page.mainFrame().evaluateJavaScript("getPrintWidth();").toInt()[0]

    def getPrintHeight(self):
        return self.__web_page.mainFrame().evaluateJavaScript("getPrintHeight();").toInt()[0]

    def getPage(self):
        return self.__web_page


    def setCustomSettings(self, settings):
        json_settings = json.dumps(settings)
        self.__web_page.mainFrame().evaluateJavaScript("setCustomSettings(%s);" % json_settings)

    def renderNow(self):
        self.__web_page.mainFrame().evaluateJavaScript("renderNow()")

    def setSettings(self, all_settings):
        settings = all_settings["settings"]
        time_min = settings["time_min"]
        time_max = settings["time_max"]
        value_min = settings["value_min"]
        value_max = settings["value_max"]
        depth_min = settings["depth_min"]
        depth_max = settings["depth_max"]
        key =  settings["report_step_time"]
        data = all_settings["data"]
        custom_settings = all_settings["custom_settings"]
        data.setParent(self)
        self.setCustomSettings(custom_settings)
        self.setScales(time_min, time_max, value_min, value_max, depth_min, depth_max)
        self.setReportStepTime(key)
        self.setPlotData(data)
        self.renderNow()








