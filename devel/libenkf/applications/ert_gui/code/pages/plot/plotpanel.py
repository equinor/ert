from PyQt4 import QtGui, QtCore
from pages.config.parameters.parameterpanel import Parameter
from pages.plot.plotview import PlotView
import ertwrapper
import pages.config.parameters.parameterpanel
import widgets.helpedwidget
from widgets.helpedwidget import ContentModel
from pages.config.parameters.parametermodels import DataModel, FieldModel, KeywordModel, SummaryModel
from pages.plot.plotdata import PlotContextDataFetcher, PlotDataFetcher, enums

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
        self.connect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.keyIndexChanged)
        self.connect(self.plotDataPanel.stateCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.stateChanged)
        parameterLayout.addWidget(self.plotList)
        parameterLayout.addWidget(self.plotDataPanel)

        self.connect(self.plotList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'), self.select)
        ContentModel.modelConnect('initialized()', self.updateList)
        #todo: listen to ensemble changes!

        plotLayout.addLayout(parameterLayout)
        plotLayout.addWidget(self.plot)
        plotLayout.addWidget(PlotViewSettingsPanel(plotView=self.plot, width=150))
        self.setLayout(plotLayout)

        self.plotDataFetcher = PlotDataFetcher()
        self.plotContextDataFetcher = PlotContextDataFetcher()


    def drawPlot(self):
        self.plot.setData(self.plotDataFetcher.data)
        self.plot.drawPlot()

    def select(self, current, previous):
        self.plotDataFetcher.setParameter(current)
        self.plotDataFetcher.setState(self.plotDataPanel.getState())
        self.plotDataFetcher.fetchContent()

        self.plotDataPanel.activatePanel(current.getType().name)

        if current.getType() == KeywordModel.TYPE:
            self.disconnect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.keyIndexChanged)
            self.plotDataPanel.keyIndexCombo.clear()
            self.plotDataPanel.keyIndexCombo.addItems(self.plotContextDataFetcher.data.getKeyIndexList(current.getName()))
            self.connect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.keyIndexChanged)

        self.drawPlot()

    def updateList(self):
        self.plotContextDataFetcher.fetchContent()
        self.plotList.clear()
        for parameter in self.plotContextDataFetcher.data.parameters:
            self.plotList.addItem(parameter)

        self.plotList.sortItems()

    def stateChanged(self):
        self.plotDataFetcher.setState(self.plotDataPanel.getState())
        self.plotDataFetcher.fetchContent()
        self.drawPlot()

    def keyIndexChanged(self, key):
        parameter = self.plotDataFetcher.getParameter()
        parameter.setData(str(key))
        self.plotDataFetcher.setState(self.plotDataPanel.getState())
        self.plotDataFetcher.fetchContent()
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

        self.showErrorbarChk = QtGui.QCheckBox("Show errorbar")
        self.connect(self.showErrorbarChk, QtCore.SIGNAL("stateChanged(int)"), lambda state : self.plotView.showErrorbar(state == QtCore.Qt.Checked))
        layout.addRow(self.showErrorbarChk)

        self.setLayout(layout)


class PlotParameterPanel(QtGui.QFrame):

    def __init__(self, parent=None, width=100):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self.setMaximumHeight(200)

        self.stack = QtGui.QStackedWidget()

        self.summaryPanel = self.createSummaryPanel()
        self.keywordPanel = self.createKeywordPanel()

        self.stack.addWidget(self.summaryPanel)
        self.stack.addWidget(self.keywordPanel)

        comboLayout, self.stateCombo = self.createStateCombo()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.stack)
        layout.addStretch()
        layout.addLayout(comboLayout)
        self.setLayout(layout)

    def activatePanel(self, parameter_type_name):
        if parameter_type_name == SummaryModel.TYPE.name:
            self.stack.setCurrentWidget(self.summaryPanel)
        elif parameter_type_name == KeywordModel.TYPE.name:
            self.stack.setCurrentWidget(self.keywordPanel)
        else:
            print "Unknown parametertype"

    def createStateCombo(self):
        layout = QtGui.QFormLayout()
        layout.setRowWrapPolicy(QtGui.QFormLayout.WrapLongRows)

        stateCombo = QtGui.QComboBox()

        for state in enums.ert_state_enum.values():
            stateCombo.addItem(state.name)
            
        stateCombo.setCurrentIndex(0)

        layout.addRow("State:", stateCombo)
        return layout, stateCombo

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

    def getState(self):
        selectedName = str(self.stateCombo.currentText())
        return enums.ert_state_enum.resolveName(selectedName)
