from PyQt4.QtCore import QSize, QSizeF, QDir, Qt
from PyQt4.QtGui import QPrinter, QImage, QPainter, QApplication, QFileDialog
from ert_gui.tools.plot.plot_bridge import PlotWebPage, PlotBridge
from ert_gui.tools.plot.plot_panel import PlotPanel


class ExportPlot(object):
    def __init__(self, active_plot_panel, time_min, time_max, value_min, value_max, depth_min, depth_max):
        super(ExportPlot, self).__init__()
        assert isinstance(active_plot_panel, PlotPanel)
        self.__active_plot_panel = active_plot_panel
        self.__time_min = time_min
        self.__time_max = time_max
        self.__value_min = value_min
        self.__value_max = value_max
        self.__depth_min = depth_min
        self.__depth_max = depth_max
        self.__bridge = None
        self.__plot_bridge_org = active_plot_panel.getPlotBridge()

        self.__width = self.__plot_bridge_org.getPrintWidth()
        self.__height = self.__plot_bridge_org.getPrintHeight() + 20


    def export(self):
        name = self.__active_plot_panel.getName()
        url = self.__active_plot_panel.getUrl()

        web_page = PlotWebPage("export - %s" % name)
        web_page.mainFrame().setScrollBarPolicy(Qt.Vertical, Qt.ScrollBarAlwaysOff)
        web_page.mainFrame().setScrollBarPolicy(Qt.Horizontal, Qt.ScrollBarAlwaysOff)

        web_page.setViewportSize(QSize(self.__width, self.__height))

        self.__bridge = PlotBridge(web_page,url)
        self.__bridge.plotReady.connect(self.plotReady)


    def plotReady(self):
        data = self.__plot_bridge_org.getPlotData()
        self.__bridge.setPlotData(data)
        self.__bridge.setScales(self.__time_min, self.__time_max, self.__value_min, self.__value_max, self.__depth_min, self.__depth_max)
        self.__bridge.updatePlotSize(QSize(self.__width, self.__height))
        self.__bridge.renderingFinished.connect(self.performExport)




    def performExport(self):
        home = QDir.homePath()
        file_name = home+"/Desktop/test.png"
            #QFileDialog.getSaveFileName(caption="Save file", directory=home,filter="Image (*.png);; PDF (*.pdf)")
        view = self.__bridge.getPage()
        if not file_name.isEmpty():
            if str(file_name).endswith(".pdf"):
                self.exportPDF(view, file_name, self.__width, self.__height)
            elif str(file_name).endswith(".png"):
                self.exportPNG(view, file_name, self.__width, self.__height)


    def exportPDF(self, view, file_name, width, height):
        pdf = QPrinter()
        pdf.setOutputFormat(QPrinter.PdfFormat)
        pdf.setPrintRange(QPrinter.AllPages)
        pdf.setOrientation(QPrinter.Portrait)
        pdf.setResolution(QPrinter.HighResolution)
        pdf.setPaperSize(QSizeF(width,height),QPrinter.Point)
        pdf.setFullPage(True)
        pdf.setOutputFileName(file_name)
        view.mainFrame().print_(pdf)


    def exportPNG(self, view, file_name, width, height):
        image = QImage(QSize(width, height), QImage.Format_ARGB32_Premultiplied)
        paint = QPainter(image)
        paint.setRenderHint(QPainter.Antialiasing, True)
        paint.setRenderHint(QPainter.HighQualityAntialiasing, True)
        paint.setRenderHint(QPainter.TextAntialiasing, True)
        paint.setRenderHint(QPainter.SmoothPixmapTransform, True)
        view.mainFrame().render(paint)

        image.save(file_name)
        paint.end()