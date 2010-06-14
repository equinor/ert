from PyQt4 import QtGui, QtCore
from pages.config.parameters.parameterpanel import Parameter
from pages.plot.plotview import PlotView
import ertwrapper
import pages.config.parameters.parameterpanel
import widgets.helpedwidget
from widgets.helpedwidget import ContentModel
from pages.config.parameters.parametermodels import DataModel, FieldModel, KeywordModel, SummaryModel
from pages.plot.plotdata import PlotContextDataFetcher, PlotDataFetcher, enums
import widgets.util
import datetime
import time
import matplotlib.dates
from pages.plot.zoomslider import ZoomSlider

class PlotPanel(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        plotLayout = QtGui.QHBoxLayout()

        self.plot = PlotView()

        parameterLayout = QtGui.QVBoxLayout()
        self.plotList = QtGui.QListWidget(self)
        self.plotList.setMaximumWidth(150)
        self.plotList.setMinimumWidth(150)

        self.plotDataPanel = PlotParameterConfigurationPanel(self, 150)
        parameterLayout.addWidget(self.plotList)
        parameterLayout.addWidget(self.plotDataPanel)

        self.connect(self.plotList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'),
                     self.select)
        ContentModel.modelConnect('initialized()', self.updateList)
        #todo: listen to ensemble changes!


        self.plotDataFetcher = PlotDataFetcher()
        self.connect(self.plotDataFetcher, QtCore.SIGNAL('dataChanged()'), self.drawPlot)
        self.plotContextDataFetcher = PlotContextDataFetcher()

        plot_view_layout = QtGui.QVBoxLayout()

        plot_view_layout.addWidget(self.plot)

        zoom_layout = QtGui.QHBoxLayout()
        self.zoom_slider = ZoomSlider()
        self.connect(self.zoom_slider, QtCore.SIGNAL('zoomValueChanged(float, float)'), self.plot.setXViewFactors)

#        self.min_value = QtGui.QDoubleSpinBox()
#        self.min_value.setMinimum(0.0)
#        self.min_value.setMaximum(1.0)
#        self.min_value.setDecimals(3)
#        self.min_value.setSingleStep(0.01)
#        self.min_value.setMaximumWidth(70)
#        self.connect(self.min_value, QtCore.SIGNAL('valueChanged(double)'), self.zoom_slider.setMinValue)
#        self.connect(self.zoom_slider, QtCore.SIGNAL('zoomValueChanged(float, float)'), lambda min, max: self.min_value.setValue(min))
#
#        self.max_value = QtGui.QDoubleSpinBox()
#        self.max_value.setValue(1.0)
#        self.max_value.setMinimum(0.0)
#        self.max_value.setMaximum(1.0)
#        self.max_value.setDecimals(3)
#        self.max_value.setSingleStep(0.01)
#        self.max_value.setMaximumWidth(70)
#        self.connect(self.max_value, QtCore.SIGNAL('valueChanged(double)'), self.zoom_slider.setMaxValue)
#        self.connect(self.zoom_slider, QtCore.SIGNAL('zoomValueChanged(float, float)'), lambda min, max: self.max_value.setValue(max))

#        zoom_layout.addWidget(self.min_value)
        zoom_layout.addWidget(self.zoom_slider)
#        zoom_layout.addWidget(self.max_value)

        plot_view_layout.addLayout(zoom_layout)

        plotLayout.addLayout(parameterLayout)
        plotLayout.addLayout(plot_view_layout)
        self.plotViewSettings = PlotViewSettingsPanel(plotView=self.plot, width=180)
        plotLayout.addWidget(self.plotViewSettings)
        self.setLayout(plotLayout)


    def drawPlot(self):
        self.plot.setData(self.plotDataFetcher.data)
        self.plot.drawPlot()

    @widgets.util.may_take_a_long_time
    def select(self, current, previous):
        self.plotDataFetcher.setParameter(current, self.plotContextDataFetcher.data)
        cw = self.plotDataFetcher.getConfigurationWidget(self.plotContextDataFetcher.data)
        self.plotDataPanel.setConfigurationWidget(cw)
        self.plotDataFetcher.fetchContent()
        self.drawPlot()

    def updateList(self):
        self.plotContextDataFetcher.fetchContent()
        self.plotList.clear()
        for parameter in self.plotContextDataFetcher.data.parameters:
            self.plotList.addItem(parameter)

        self.plotList.sortItems()

        self.plotViewSettings.setDefaultErrorbarMaxValue(self.plotContextDataFetcher.data.errorbar_max)
        self.plot.setPlotPath(self.plotContextDataFetcher.data.plot_path)


class PlotViewSettingsPanel(QtGui.QFrame):
    def __init__(self, parent=None, plotView=None, width=100):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.setMinimumWidth(width)
        self.setMaximumWidth(width)

        self.plotView = plotView

        layout = QtGui.QFormLayout()
        layout.setRowWrapPolicy(QtGui.QFormLayout.WrapLongRows)


        layout.addRow("Line style:", self.createPlotLineStyleLayout(self.plotView.setPlotType))



        self.errorbarModes = QtGui.QComboBox()
        errorbarItems = ["Off", "Auto", "Errorbar", "Errorline", "History"]
        self.errorbarModes.addItems(errorbarItems)
        self.errorbar_max = -1
        def errorbar(index):
            if index == 0: #off
                self.plotView.showErrorbar(False, True)
            elif index == 1: #auto
                self.plotView.showErrorbar(True, True)
                self.plotView.setErrorbarLimit(self.errorbar_max)
            elif index == 2: #only show show errorbars
                self.plotView.showErrorbar(True, True)
                self.plotView.setErrorbarLimit(10000)
            elif index == 3: #only show error lines with standard deviation
                self.plotView.showErrorbar(True, True)
                self.plotView.setErrorbarLimit(0)
                self.plotView.setShowSTDV(True)
            else: #only show error lines without standard deviation
                self.plotView.showErrorbar(True, True)
                self.plotView.setErrorbarLimit(0)
                self.plotView.setShowSTDV(False)


        self.connect(self.errorbarModes, QtCore.SIGNAL("currentIndexChanged(int)"), errorbar)
        layout.addRow("Error and history:", self.errorbarModes)

        layout.addRow("Observation line style:", self.createPlotLineStyleLayout(self.plotView.setObservationPlotType))

        self.alphaSpn = QtGui.QDoubleSpinBox(self)
        self.alphaSpn.setMinimum(0.0)
        self.alphaSpn.setMaximum(1.0)
        self.alphaSpn.setDecimals(3)
        self.alphaSpn.setSingleStep(0.01)
        self.alphaSpn.setValue(plotView.getAlphaValue())
        self.connect(self.alphaSpn, QtCore.SIGNAL('valueChanged(double)'), self.plotView.setAlphaValue)
        layout.addRow("Blend factor:", self.alphaSpn)

        layout.addRow(widgets.util.createSeparator())

        self.saveBtn = QtGui.QPushButton()
        self.saveBtn.setIcon(widgets.util.resourceIcon("disk"))
        self.saveBtn.setIconSize(QtCore.QSize(16, 16))
        layout.addRow("Save:", self.saveBtn)
        self.connect(self.saveBtn, QtCore.SIGNAL('clicked()'), self.plotView.save)

        layout.addRow(widgets.util.createSeparator())

        self.selected_member_label = QtGui.QLabel()
        self.selected_member_label.setWordWrap(True)
        def plotSelectionChanged(selected_members):
            text = ""
            for member in selected_members:
                text = text + " " + str(member.get_gid())
            self.selected_member_label.setText(text)

        self.connect(self.plotView, QtCore.SIGNAL('plotSelectionChanged(array)'), plotSelectionChanged)
        layout.addRow(QtGui.QLabel("Selected members:"))
        layout.addRow(self.selected_member_label)

        self.clear_button = QtGui.QPushButton()
        self.clear_button.setText("Clear selection")
        layout.addRow(self.clear_button)
        self.connect(self.clear_button, QtCore.SIGNAL('clicked()'), self.plotView.clearSelection)

        self.setLayout(layout)


    def setDefaultErrorbarMaxValue(self, errorbar_max):
        self.errorbar_max = errorbar_max
        if self.errorbarModes.currentIndex == 0: #auto
            self.plotView.setErrorbarLimit(errorbar_max)


    def createPlotLineStyleLayout(self, func):
        plot_style_layout = QtGui.QHBoxLayout()

        plot_marker_styles = ["", ".", ",", "o", "*", "s", "+", "x", "p", "h", "H", "D", "d"]
        plot_line_styles = ["", "-", "--", "-.", ":"]

        plot_marker_type = QtGui.QComboBox()
        plot_line_type = QtGui.QComboBox()

        plot_marker_type.addItems(plot_marker_styles)
        plot_line_type.addItems(plot_line_styles)

        def combinePlotTypes():
            res = str(plot_marker_type.currentText()) + str(plot_line_type.currentText())
            func(res)

        self.connect(plot_marker_type, QtCore.SIGNAL("currentIndexChanged(QString)"), combinePlotTypes)
        self.connect(plot_line_type, QtCore.SIGNAL("currentIndexChanged(QString)"), combinePlotTypes)

        plot_style_layout.addWidget(plot_marker_type)
        plot_style_layout.addWidget(plot_line_type)

        return plot_style_layout


class PlotParameterConfigurationPanel(QtGui.QFrame):
    def __init__(self, parent=None, width=100):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self.setMaximumHeight(200)

        self.layout = QtGui.QStackedLayout()
        self.setLayout(self.layout)

    def setConfigurationWidget(self, widget):
        if self.layout.indexOf(widget) == -1:
            self.layout.addWidget(widget)
        self.layout.setCurrentWidget(widget)
