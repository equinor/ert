from PyQt4 import QtGui, QtCore
from pages.config.parameters.parameterpanel import Parameter
from pages.plot.plotview import PlotView
import ertwrapper
import pages.config.parameters.parameterpanel

class PlotPanel(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        plotLayout = QtGui.QHBoxLayout()

        self.plot = PlotView()

        parameterLayout = QtGui.QVBoxLayout()
        self.plotList = QtGui.QListWidget(self)
        self.plotList.setMaximumWidth(150)
        self.plotList.setMinimumWidth(150)

        self.plotDataPanel = ParameterPlotPanel(self, 150)
        self.connect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.keyIndexChanged)
        parameterLayout.addWidget(self.plotList)
        parameterLayout.addWidget(self.plotDataPanel)

        self.connect(self.plotList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'), self.select)
        ContentModel.modelConnect('initialized()', self.updateList)
        #todo: listen to ensemble changes!

        plotLayout.addLayout(parameterLayout)
        plotLayout.addWidget(self.plot)
        self.setLayout(plotLayout)


    def select(self, current, previous):
        self.plot.plotDataFetcher.setParameter(current)
        self.plot.plotDataFetcher.fetchContent()

        self.disconnect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.keyIndexChanged)
        self.plotDataPanel.keyIndexCombo.clear()
        self.plotDataPanel.keyIndexCombo.addItems(self.plot.plotContextDataFetcher.data.getKeyIndexList(current.getName()))
        self.connect(self.plotDataPanel.keyIndexCombo, QtCore.SIGNAL('currentIndexChanged(QString)'), self.keyIndexChanged)

        self.plot.drawPlot()

    def updateList(self):
        self.plot.plotContextDataFetcher.fetchContent()
        self.plotList.clear()
        for parameter in self.plot.plotContextDataFetcher.data.parameters:
            self.plotList.addItem(parameter)

        self.plotList.sortItems()

    def keyIndexChanged(self, key):
        parameter = self.plot.plotDataFetcher.getParameter()
        parameter.setData(str(key))
        self.plot.plotDataFetcher.fetchContent()
        self.plot.drawPlot()


class ParameterPlotPanel(QtGui.QStackedWidget):

    def __init__(self, parent=None, width=100):
        QtGui.QStackedWidget.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self.setMaximumHeight(100)

        self.addWidget(self.createPanel())

    def createPanel(self):
        panel = QtGui.QFrame()
        panel.setFrameShape(QtGui.QFrame.StyledPanel)
        panel.setFrameShadow(QtGui.QFrame.Plain)
        panel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        layout = QtGui.QFormLayout()
        layout.setRowWrapPolicy(QtGui.QFormLayout.WrapLongRows)
        self.keyIndexCombo = QtGui.QComboBox()

        layout.addRow("Key index:", self.keyIndexCombo)
        panel.setLayout(layout)
        return panel


