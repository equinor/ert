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
from widgets.configpanel import ConfigPanel
from PyQt4.Qt import SIGNAL
from PyQt4.QtCore import QDate, Qt, QPoint
from pages.plot.plotconfig import PlotConfigPanel
from PyQt4.QtGui import QTabWidget, QFormLayout, QFrame, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QToolButton, QMainWindow
from PyQt4.QtGui import QCalendarWidget

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

        plot_view_layout = QtGui.QGridLayout()

        plot_view_layout.addWidget(self.plot, 0, 0, 1, 1)

        self.h_zoom_slider = ZoomSlider()
        self.connect(self.h_zoom_slider, QtCore.SIGNAL('zoomValueChanged(float, float)'), self.plot.setXViewFactors)

        self.v_zoom_slider = ZoomSlider(horizontal=False)
        self.connect(self.v_zoom_slider, QtCore.SIGNAL('zoomValueChanged(float, float)'), self.plot.setYViewFactors)

        plot_view_layout.addWidget(self.h_zoom_slider, 1, 0, 1, 1)
        plot_view_layout.addWidget(self.v_zoom_slider, 0, 1, 1, 1)

        plotLayout.addLayout(parameterLayout)
        plotLayout.addLayout(plot_view_layout)

        self.plotViewSettings = PlotViewSettingsPanel(plotView=self.plot, width=250)
        plotLayout.addWidget(self.plotViewSettings)
        
        self.setLayout(plotLayout)


    def drawPlot(self):
        data = self.plotDataFetcher.data

        self.plot.setData(data)

        x_min = self.plot.plot_settings.getMinXLimit(data.x_min, data.getXDataType())
        x_max = self.plot.plot_settings.getMaxXLimit(data.x_max, data.getXDataType())
        y_min = self.plot.plot_settings.getMinYLimit(data.y_min, data.getYDataType())
        y_max = self.plot.plot_settings.getMaxYLimit(data.y_max, data.getYDataType())

#        x_min = data.x_min
#        x_max = data.x_max
#
#        if data.getXDataType() == "time" and not x_min is None and not x_max is None:
#            x_min = x_min.value / 86400.0
#            x_max = x_max.value / 86400.0
#
#        y_min = data.y_min
#        y_max = data.y_max
#
#        if data.getYDataType() == "time" and not y_min is None and not y_max is None:
#            y_min = y_min.value / 86400.0
#            y_max = y_max.value / 86400.0

        self.plotViewSettings.setLimits(x_min, x_max, y_min, y_max)
        self.plotViewSettings.setLimitStates(*self.plot.plot_settings.getLimitStates())
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

        self.plot.setPlotPath(self.plotContextDataFetcher.data.plot_path)
        self.plot.setPlotConfigPath(self.plotContextDataFetcher.data.plot_config_path)


class PlotViewSettingsPanel(QtGui.QFrame):

    def __init__(self, parent=None, plotView=None, width=100):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.setMinimumWidth(width)
        self.setMaximumWidth(width)

        self.plotView = plotView

        layout = QtGui.QVBoxLayout()

        plot_configs = self.plotView.getPlotConfigList()
        tabbed_panel = QTabWidget()
        tabbed_panel.setTabPosition(QTabWidget.West)
        for plot_config in plot_configs:
            config_panel = PlotConfigPanel(plot_config)
            tabbed_panel.addTab(config_panel, plot_config.name)
            self.connect(config_panel, SIGNAL('plotConfigChanged()'), self.plotView.drawPlot)

        layout.addWidget(tabbed_panel)

        layout.addWidget(self.createMemberSelectionPanel())
        layout.addWidget(widgets.util.createSeparator())
        layout.addWidget(self.createPlotRangePanel())
        layout.addWidget(widgets.util.createSeparator())

        layout.addLayout(self.createSaveButtonLayout())

        self.setLayout(layout)

    def setLimits(self, x_min, x_max, y_min, y_max):
        self.updateSpinner(self.x_min_spinner, x_min)
        self.updateSpinner(self.x_max_spinner, x_max)
        self.updateSpinner(self.y_min_spinner, y_min)
        self.updateSpinner(self.y_max_spinner, y_max)

    def setLimitStates(self, x_min_state, x_max_state, y_min_state, y_max_state):
        self.x_min_check.setChecked(x_min_state)
        self.x_max_check.setChecked(x_max_state)
        self.y_min_check.setChecked(y_min_state)
        self.y_max_check.setChecked(y_max_state)

    def updateSpinner(self, spinner, value):
        if not value is None:
            state = spinner.blockSignals(True)
            spinner.setValue(value)
            spinner.blockSignals(state)


    def createDisableableSpinner(self, func):
        layout = QHBoxLayout()

        check = QCheckBox()

        spinner = QtGui.QDoubleSpinBox()
        spinner.setDecimals(3)
        spinner.setSingleStep(1)
        spinner.setDisabled(True)
        spinner.setMinimum(-10000000000)
        spinner.setMaximum(10000000000)
        spinner.setMaximumWidth(100)
        self.connect(spinner, QtCore.SIGNAL('valueChanged(double)'), func)

        popup_button = QToolButton()
        popup_button.setIcon(widgets.util.resourceIcon("calendar"))
        popup_button.setIconSize(QtCore.QSize(16, 16))
        popup_button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        popup_button.setAutoRaise(True)
        popup_button.setDisabled(True)

        def popup():
            popup = Popup(popup_button, spinner.value())
            self.connect(popup, QtCore.SIGNAL('dateChanged(double)'), lambda date : spinner.setValue(date))

        self.connect(popup_button, QtCore.SIGNAL('clicked()'), popup)

        def disabler(state):
            disabled = not state == 2
            spinner.setDisabled(disabled)
            popup_button.setDisabled(disabled)

            if not disabled:
                func(spinner.value())
            else:
                func(None)

        self.connect(check, SIGNAL('stateChanged(int)'), disabler)

        layout.addWidget(check)
        layout.addWidget(spinner)
        layout.addWidget(popup_button)
        return layout, spinner, check

    def createPlotRangePanel(self):
        frame = QFrame()
        frame.setMaximumHeight(150)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Plain)

        layout = QFormLayout()

        x_min_layout, self.x_min_spinner, self.x_min_check = self.createDisableableSpinner(self.plotView.setMinXLimit)
        x_max_layout, self.x_max_spinner, self.x_max_check  = self.createDisableableSpinner(self.plotView.setMaxXLimit)

        layout.addRow("X min:", x_min_layout)
        layout.addRow("X max:", x_max_layout)

        y_min_layout, self.y_min_spinner, self.y_min_check  = self.createDisableableSpinner(self.plotView.setMinYLimit)
        y_max_layout, self.y_max_spinner, self.y_max_check = self.createDisableableSpinner(self.plotView.setMaxYLimit)

        layout.addRow("Y min:", y_min_layout)
        layout.addRow("Y max:", y_max_layout)

        frame.setLayout(layout)
        return frame

    def createSaveButtonLayout(self):
        save_layout = QFormLayout()
        save_layout.setRowWrapPolicy(QtGui.QFormLayout.WrapLongRows)

        self.save_button = QtGui.QPushButton()
        self.save_button.setIcon(widgets.util.resourceIcon("disk"))
        self.save_button.setIconSize(QtCore.QSize(16, 16))

        save_layout.addRow("Save:", self.save_button)
        self.connect(self.save_button, QtCore.SIGNAL('clicked()'), self.plotView.save)

        return save_layout

    def createMemberSelectionPanel(self):
        frame = QFrame()
        frame.setMinimumHeight(100)
        frame.setMaximumHeight(100)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Plain)

        layout = QVBoxLayout()

        self.selected_member_label = QtGui.QLabel()
        self.selected_member_label.setWordWrap(True)

        def plotSelectionChanged(selected_members):
            text = ""
            for member in selected_members:
                text = text + " " + str(member)
            self.selected_member_label.setText(text)

        self.connect(self.plotView, QtCore.SIGNAL('plotSelectionChanged(array)'), plotSelectionChanged)
        
        layout.addWidget(QtGui.QLabel("Selected members:"))
        layout.addWidget(self.selected_member_label)

        layout.addStretch(1)

        self.clear_button = QtGui.QPushButton()
        self.clear_button.setText("Clear selection")
        layout.addWidget(self.clear_button)
        self.connect(self.clear_button, QtCore.SIGNAL('clicked()'), self.plotView.clearSelection)
        frame.setLayout(layout)

        return frame



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

class Popup(QFrame):
    def __init__(self, parent=None, date_in_days=0):
        QFrame.__init__(self, parent, Qt.Popup | Qt.Window | Qt.FramelessWindowHint)
        layout = QVBoxLayout()

        if not parent is None:
            p = parent.mapToGlobal(QPoint(0,0))
            self.setGeometry(p.x(), p.y(), 10, 10)
            
        self.calendar = QCalendarWidget()
        layout.addWidget(self.calendar)
        self.setLayout(layout)

        self.setDateFromDays(date_in_days)
        self.connect(self.calendar, QtCore.SIGNAL('selectionChanged()'), self.finished)
        self.show()

    def finished(self):
        date = self.calendar.selectedDate()
        self.emit(SIGNAL('dateChanged(double)'), self.qDateToDays(date))
        self.close()

    def focusOutEvent(self, event):
        self.finished()

    def setDateFromDays(self, days):
        t = time.localtime(days * 86400.0)
        self.calendar.setSelectedDate(QDate(*t[0:3]))

    def qDateToDays(self, qdate):
        date = qdate.toPyDate()
        seconds = int(time.mktime(date.timetuple()))
        return seconds / 86400.0

