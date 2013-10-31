import json
import os
from PyQt4.QtCore import QUrl, Qt, QObject, pyqtSignal, pyqtSlot, QVariant
from PyQt4.QtGui import QWidget, QGridLayout, QPainter
from PyQt4.QtNetwork import QNetworkProxy
from PyQt4.QtWebKit import QWebView, QWebPage, QWebSettings
from ert_gui.models.connectors.plot.ensemble_summary_plot import EnsembleSummaryPlot


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


class PlotContextObject(QObject):
    def __init__(self, data, parent=None):
        QObject.__init__(self, parent)
        self.__data = data

    @pyqtSlot(result=str)
    def getPlotData(self):
        return json.dumps(EnsembleSummaryPlot().getPlotData())



class PlotPanel(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        proxy = QNetworkProxy(QNetworkProxy.HttpProxy, "www-proxy.statoil.no", 80)
        QNetworkProxy.setApplicationProxy(proxy)


        root_path = os.getenv("ERT_SHARE_PATH")
        path = os.path.join(root_path, "gui/plots/plot.html")
        # print(path)

        layout = QGridLayout()

        self.web_view = PlotWebView()
        self.web_view.page().mainFrame().javaScriptWindowObjectCleared.connect(self.applyContextObject)
        self.web_view.setUrl(QUrl("file://%s" % path))

        self.context_object = PlotContextObject(EnsembleSummaryPlot().getPlotData(), self)
        self.applyContextObject()

        layout.addWidget(self.web_view)

        self.setLayout(layout)


        # print(json.dumps(EnsembleSummaryPlot().getPlotData()))

    def applyContextObject(self):
       self.web_view.page().mainFrame().addToJavaScriptWindowObject("plot_data_source", self.context_object)


    def getName(self):
        return "Plot"



