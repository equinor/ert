from PyQt4.QtCore import QSize, QSizeF, QDir, Qt, QStringList, QString
from PyQt4.QtGui import QPrinter, QImage, QPainter, QApplication, QFileDialog
from PyQt4.QtWebKit import QWebPage
from ert_gui.tools.plot.plot_bridge import PlotWebPage, PlotBridge
from ert_gui.tools.plot.plot_panel import PlotPanel


class ExportPlot(object):
    def __init__(self, active_plot_panel, settings):
        super(ExportPlot, self).__init__()
        assert isinstance(active_plot_panel, PlotPanel)
        self.__active_plot_panel = active_plot_panel
        self.__time_min = settings["time_min"]
        self.__time_max = settings["time_max"]
        self.__value_min = settings["value_min"]
        self.__value_max = settings["value_max"]
        self.__depth_min = settings["depth_min"]
        self.__depth_max = settings["depth_max"]
        self.__report_step_time = settings["report_step_time"]
        self.__custom_settings = settings["custom_settings"]
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
        self.__bridge.setCustomSettings(self.__custom_settings)
        self.__bridge.setReportStepTime(self.__report_step_time)
        self.__bridge.renderingFinished.connect(self.performExport)




    def performExport(self):
        home = QDir.homePath()
        dialog = QFileDialog(self.__active_plot_panel.parent())
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter("Image (*.png);; PDF (*.pdf)")
        dialog.setWindowTitle("Export plot")
        dialog.setDirectory(home)
        file_type_label = QString("Select file type: ")
        dialog.setLabelText(QFileDialog.FileType, file_type_label)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        if dialog.exec_():
            result = dialog.selectedFiles()
            assert isinstance(result, QStringList)
            if len(result) == 1:
                file_name = result[0]
                selected_file_type = dialog.selectedNameFilter()
                view = self.__bridge.getPage()
                if not file_name.isEmpty():
                    if str(selected_file_type) == "PDF (*.pdf)":
                        if not str(file_name).endswith(".pdf"):
                            file_name=str(file_name) + ".pdf"
                        self.exportPDF(view, file_name, self.__width, self.__height)
                    elif str(selected_file_type) == "Image (*.png)":
                        if not str(file_name).endswith(".png"):
                            file_name=str(file_name) + ".png"
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