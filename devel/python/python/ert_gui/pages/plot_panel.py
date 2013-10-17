import json
import os
from PyQt4.QtCore import QUrl, Qt
from PyQt4.QtGui import QWidget, QGridLayout, QPainter
from PyQt4.QtNetwork import QNetworkProxy
from PyQt4.QtWebKit import QWebView, QWebPage
from ert_gui.models.connectors.plot.observations import ObservationsModel


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


class PlotPanel(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        proxy = QNetworkProxy(QNetworkProxy.HttpProxy, "www-proxy.statoil.no", 80)
        QNetworkProxy.setApplicationProxy(proxy)


        root_path = os.getenv("ERT_SHARE_PATH")
        path = os.path.join(root_path, "gui/plots/plot.html")
        # print(path)

        layout = QGridLayout()

        web_view = PlotWebView()
        web_view.setUrl(QUrl("file://%s" % path))
        # web_view.page().mainFrame().setScrollBarPolicy(Qt.Horizontal, Qt.ScrollBarAlwaysOff)
        # web_view.page().mainFrame().setScrollBarPolicy(Qt.Vertical, Qt.ScrollBarAlwaysOff)
        web_view.show()
        layout.addWidget(web_view)

        self.setLayout(layout)

        # print(json.dumps(ObservationsModel().getAllObservations()))



    def getName(self):
        return "Plot"




