# Copyright (C) 2014 Statoil ASA, Norway.
#
# The file 'export_plot.py' is part of ERT - Ensemble based Reservoir Tool.
#
# ERT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ERT is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
# for more details.


from PyQt4.QtCore import QSize, QSizeF, Qt, QStringList
from PyQt4.QtGui import QPrinter, QImage, QPainter, QFileDialog

from ert.util import ctime
from ert_gui.models.connectors.plot.plot_settings import PlotSettingsModel
from ert_gui.tools.plot.plot_bridge import PlotWebPage, PlotBridge
from ert_gui.tools.plot.plot_panel import PlotPanel

class ExportPlot(object):
    def __init__(self, active_plot_panel, plot_settings):
        super(ExportPlot, self).__init__()
        assert isinstance(active_plot_panel, PlotPanel)
        settings = plot_settings["settings"]
        self.__active_plot_panel = active_plot_panel
        self.__time_min = settings["time_min"]
        self.__time_max = settings["time_max"]
        self.__value_min = settings["value_min"]
        self.__value_max = settings["value_max"]
        self.__depth_min = settings["depth_min"]
        self.__depth_max = settings["depth_max"]
        self.__report_step_time = settings["report_step_time"]
        self.__custom_settings = plot_settings["custom_settings"]
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
        default_export_path = PlotSettingsModel().getDefaultPlotPath()



        #home = QDir.homePath()
        dialog = QFileDialog(self.__active_plot_panel.parent())
        dialog.setFileMode(QFileDialog.AnyFile)
        #dialog.setNameFilter("Image (*.png);; PDF (*.pdf)")
        dialog.setNameFilter("Image (*.png)")
        dialog.setWindowTitle("Export plot")
        dialog.setDirectory(default_export_path)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        #file_type_label = QString("Select file type: ")
        dialog.setLabelText(QFileDialog.FileType, "Select file type: ")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        default_file_name = self.getDefaultFileName()
        dialog.selectFile(default_file_name)
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

    def getDefaultFileName(self):
        name = self.__plot_bridge_org.getPlotData().name()
        name = name.replace(":"," ")
        type = self.__active_plot_panel.getName()
        if type == "Histogram":
            time = ctime(self.__report_step_time)
            name = "%s - %s" % (name, time.date())

        name = type +" "+name

        return name