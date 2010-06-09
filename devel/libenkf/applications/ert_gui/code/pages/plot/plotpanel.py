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

class PlotPanel(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        plotLayout = QtGui.QHBoxLayout()

        self.plot = PlotView()

        parameterLayout = QtGui.QVBoxLayout()
        self.plotList = QtGui.QListWidget(self)
        self.plotList.setMaximumWidth(150)
        self.plotList.setMinimumWidth(150)

        self.plotDataPanel = PlotParameterPanel(self, 150)
        self.connect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.parameterChanged)
        self.connect(self.plotDataPanel.stateCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.parameterChanged)
        self.connect(self.plotDataPanel.keyIndexI, QtCore.SIGNAL('valueChanged(int)'), self.parameterChanged)
        self.connect(self.plotDataPanel.keyIndexJ, QtCore.SIGNAL('valueChanged(int)'), self.parameterChanged)
        self.connect(self.plotDataPanel.keyIndexK, QtCore.SIGNAL('valueChanged(int)'), self.parameterChanged)
        parameterLayout.addWidget(self.plotList)
        parameterLayout.addWidget(self.plotDataPanel)

        self.connect(self.plotList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'), self.select)
        ContentModel.modelConnect('initialized()', self.updateList)
        #todo: listen to ensemble changes!


        self.plotDataFetcher = PlotDataFetcher()
        self.plotContextDataFetcher = PlotContextDataFetcher()

        plot_view_layout = QtGui.QVBoxLayout()

        self.xlimitLower = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.xlimitUpper = QtGui.QSlider(QtCore.Qt.Horizontal)

        self.xlimitUpper.setValue(self.xlimitUpper.maximum())

        self.xlimitUpper.connect(self.xlimitUpper, QtCore.SIGNAL('valueChanged(int)'), self.xlimitLower.setMaximum)
        self.xlimitUpper.connect(self.xlimitUpper, QtCore.SIGNAL('valueChanged(int)'), self.rangeChanged)
        self.xlimitUpper.connect(self.xlimitLower, QtCore.SIGNAL('valueChanged(int)'), self.xlimitUpper.setMinimum)
        self.xlimitUpper.connect(self.xlimitLower, QtCore.SIGNAL('valueChanged(int)'), self.rangeChanged)

        plot_view_layout.addWidget(self.plot)
        plot_view_layout.addWidget(self.xlimitLower)
        plot_view_layout.addWidget(self.xlimitUpper)


        plotLayout.addLayout(parameterLayout)
        plotLayout.addLayout(plot_view_layout)
        self.plotViewSettings = PlotViewSettingsPanel(plotView=self.plot, width=180)
        plotLayout.addWidget(self.plotViewSettings)
        self.setLayout(plotLayout)


    def drawPlot(self):
        self.plot.setData(self.plotDataFetcher.data)
        self.plot.drawPlot()

    def rangeChanged(self):
        xminf = self.xlimitLower.value() / (self.xlimitUpper.maximum() * 1.0)
        xmaxf = self.xlimitUpper.value() / (self.xlimitUpper.maximum() * 1.0)
        self.plot.setXViewFactors(xminf, xmaxf)


    def select(self, current, previous):

        if current.getType() == KeywordModel.TYPE:
            self.disconnect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.parameterChanged)
            self.plotDataPanel.keyIndexCombo.clear()
            self.plotDataPanel.keyIndexCombo.addItems(self.plotContextDataFetcher.data.getKeyIndexList(current.getName()))
            self.connect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.parameterChanged)
        elif current.getType() == FieldModel.TYPE:
            bounds = self.plotContextDataFetcher.data.field_bounds
            self.plotDataPanel.setFieldBounds(*bounds)

        self.plotDataFetcher.setParameter(current)
        self.initParameter()
        self.plotDataPanel.activatePanel(current.getType())
        self.drawPlot()

    def updateList(self):
        self.plotContextDataFetcher.fetchContent()
        self.plotList.clear()
        for parameter in self.plotContextDataFetcher.data.parameters:
            self.plotList.addItem(parameter)

        self.plotList.sortItems()

        self.plotViewSettings.setDefaultErrorbarMaxValue(self.plotContextDataFetcher.data.errorbar_max)
        self.plot.setPlotPath(self.plotContextDataFetcher.data.plot_path)

    @widgets.util.may_take_a_long_time
    def initParameter(self):
        parameter = self.plotDataFetcher.getParameter()
        self.plotDataFetcher.setState(self.plotDataPanel.getState())

        if parameter.getType() == FieldModel.TYPE:
            position = self.plotDataPanel.getFieldPosition()
            parameter.setUserData(position)
        elif parameter.getType() == KeywordModel.TYPE:
            parameter.setUserData(self.plotDataPanel.getKeyIndex())
        else:
            parameter.setUserData(None)

        self.plotDataFetcher.fetchContent()

    def parameterChanged(self):
        self.initParameter()
        self.drawPlot()


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


class PlotParameterPanel(QtGui.QFrame):

    def __init__(self, parent=None, width=100):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self.setMaximumHeight(200)

#        self.layout = QtGui.QGridLayout()
#        self.setLayout(self.layout)

        self.stack = QtGui.QStackedWidget()

        self.empty_panel = QtGui.QFrame()
        self.summaryPanel = self.createSummaryPanel()
        self.keywordPanel = self.createKeywordPanel()
        self.fieldPanel = self.createFieldPanel()

        self.stack.addWidget(self.empty_panel)
        self.stack.addWidget(self.summaryPanel)
        self.stack.addWidget(self.keywordPanel)
        self.stack.addWidget(self.fieldPanel)

        self.combo_widget, self.stateCombo = self.createStateCombo()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.stack)
        layout.addStretch()
        layout.addWidget(self.combo_widget)
        self.setLayout(layout)
        self.activatePanel(None)

    def setWidget(self, widget):
        self.layout.addWidget(widget, 0, 0)

    def activatePanel(self, parameter_type):
        self.combo_widget.setHidden(False)

        if parameter_type == SummaryModel.TYPE:
            self.stack.setCurrentWidget(self.summaryPanel)
        elif parameter_type == KeywordModel.TYPE:
            self.stack.setCurrentWidget(self.keywordPanel)
        elif parameter_type == FieldModel.TYPE:
            self.stack.setCurrentWidget(self.fieldPanel)
        else:
            self.combo_widget.setHidden(True)
            self.stack.setCurrentWidget(self.empty_panel)
            print "No panel for this parametertype"

    def createStateCombo(self):
        widget = QtGui.QFrame()
        layout = QtGui.QFormLayout()
        layout.setMargin(0)
        layout.setRowWrapPolicy(QtGui.QFormLayout.WrapLongRows)

        stateCombo = QtGui.QComboBox()

        for state in enums.ert_state_enum.values():
            stateCombo.addItem(state.name)

        stateCombo.setCurrentIndex(0)

        layout.addRow("State:", stateCombo)
        widget.setLayout(layout)
        return widget, stateCombo

    def createSummaryPanel(self):
        panel = QtGui.QFrame()
        panel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        return panel

    def createKeywordPanel(self):
        widget = QtGui.QWidget()
        layout = QtGui.QFormLayout()
        layout.setMargin(0)
        layout.setRowWrapPolicy(QtGui.QFormLayout.WrapLongRows)
        self.keyIndexCombo = QtGui.QComboBox()
        layout.addRow("Key index:", self.keyIndexCombo)
        widget.setLayout(layout)
        return widget

    def createFieldPanel(self):
        widget = QtGui.QWidget()
        layout = QtGui.QFormLayout()
        layout.setMargin(0)
        layout.setRowWrapPolicy(QtGui.QFormLayout.WrapLongRows)

        self.keyIndexI = QtGui.QSpinBox()
        self.keyIndexI.setMinimum(1)

        layout.addRow("i:", self.keyIndexI)

        self.keyIndexJ = QtGui.QSpinBox()
        self.keyIndexJ.setMinimum(1)
        layout.addRow("j:", self.keyIndexJ)

        self.keyIndexK = QtGui.QSpinBox()
        self.keyIndexK.setMinimum(1)
        layout.addRow("k:", self.keyIndexK)

        self.setFieldBounds(1, 1, 1)

        widget.setLayout(layout)
        return widget

    def setFieldBounds(self, i, j, k):
        self.keyIndexI.setMaximum(i)
        self.keyIndexJ.setMaximum(j)
        self.keyIndexK.setMaximum(k)

    def getFieldPosition(self):
        return (self.keyIndexI.value(), self.keyIndexJ.value(), self.keyIndexK.value())

    def getState(self):
        selectedName = str(self.stateCombo.currentText())
        return enums.ert_state_enum.resolveName(selectedName)

    def getKeyIndex(self):
        return str(self.keyIndexCombo.currentText())
