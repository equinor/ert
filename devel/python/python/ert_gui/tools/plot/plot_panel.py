import json
import os
from PyQt4.QtCore import QUrl, Qt, pyqtSlot, pyqtSignal
from PyQt4.QtGui import QWidget, QGridLayout, QPainter
from PyQt4.QtWebKit import QWebView, QWebPage, QWebSettings


class PlotWebPage(QWebPage):
    def __init__(self):
        QWebPage.__init__(self)

    def javaScriptConsoleMessage(self, message, line_number, source_id):
        print("Source: %s at line: %d -> %s" % (source_id, line_number, message))


class PlotWebView(QWebView):
    def __init__(self):
        QWebView.__init__(self)
        self.setPage(PlotWebPage())
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.settings().setAttribute(QWebSettings.JavascriptEnabled, True)
        self.settings().setAttribute(QWebSettings.LocalContentCanAccessFileUrls, True)
        self.settings().setAttribute(QWebSettings.LocalContentCanAccessRemoteUrls, True)



class PlotPanel(QWidget):
    plotReady = pyqtSignal()

    def __init__(self, plot_url):
        QWidget.__init__(self)

        self.__data = []
        root_path = os.getenv("ERT_SHARE_PATH")
        path = os.path.join(root_path, plot_url)

        layout = QGridLayout()

        self.web_view = PlotWebView()
        self.web_view.page().mainFrame().javaScriptWindowObjectCleared.connect(self.applyContextObject)
        self.web_view.setUrl(QUrl("file://%s" % path))
        self.web_view.loadFinished.connect(self.loadFinished)

        self.applyContextObject()

        layout.addWidget(self.web_view)

        self.setLayout(layout)


    @pyqtSlot(result=str)
    def getPlotData(self):
        return json.dumps(self.__data)


    def setPlotData(self, data):
        self.__data = data
        self.web_view.page().mainFrame().evaluateJavaScript("updatePlot();")


    def applyContextObject(self):
       self.web_view.page().mainFrame().addToJavaScriptWindowObject("plot_data_source", self)


    def loadFinished(self, ok):
        self.plotReady.emit()




