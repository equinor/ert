from PyQt4 import QtGui, QtCore
import ertwrapper

from widgets.helpedwidget import HelpedWidget, ContentModel
from widgets.util import resourceIcon, ListCheckPanel, ValidatedTimestepCombo, createSpace, getItemsFromList, frange
import threading
import time
import widgets
import math

class SimulationList(QtGui.QListWidget):
    def __init__(self):
        QtGui.QListWidget.__init__(self)

        self.setViewMode(QtGui.QListView.IconMode)
        self.setMovement(QtGui.QListView.Static)
        self.setResizeMode(QtGui.QListView.Adjust)

        self.setItemDelegate(SimulationItemDelegate())

class SimulationItem(QtGui.QListWidgetItem):
    def __init__(self, simulation):
        self.simulation = simulation
        QtGui.QListWidgetItem.__init__(self, type=9901)
        self.updateSimulation()

    def updateSimulation(self):
        self.setData(QtCore.Qt.DisplayRole, self.simulation)

    def __ge__(self, other):
        return self.simulation.name >= other.simulation.name

    def __lt__(self, other):
        return not self >= other

class SimulationItemDelegate(QtGui.QStyledItemDelegate):
    waiting = QtGui.QColor(200, 200, 255)
    running = QtGui.QColor(200, 255, 200)
    failed = QtGui.QColor(255, 200, 200)
    unknown = QtGui.QColor(255, 200, 128)
    finished = QtGui.QColor(200, 200, 200)
    notactive = QtGui.QColor(255, 255, 255)

    size = QtCore.QSize(32, 27)

    def __init__(self):
        QtGui.QStyledItemDelegate.__init__(self)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        data = index.model().data(index)

        if data is None:
            data = Simulation("0")
            data.status = 0
        else:
            data = data.toPyObject()

        if data.isWaiting():
            color = self.waiting
        elif data.isRunning():
            color = self.running
        elif data.finishedSuccesfully():
            color = self.finished
        elif data.hasFailed():
            color = self.failed
        elif data.notActive():
            color = self.notactive
        else:
            color = self.unknown

        painter.setPen(color)
        rect = QtCore.QRect(option.rect)
        rect.setX(rect.x() + 1)
        rect.setY(rect.y() + 1)
        rect.setWidth(rect.width() - 2)
        rect.setHeight(rect.height() - 2)
        painter.fillRect(rect, color)


        
        painter.setPen(QtCore.Qt.black)
        painter.drawText(rect, QtCore.Qt.AlignCenter + QtCore.Qt.AlignVCenter, str(data.name))

        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        painter.drawRect(rect)

        prect = QtCore.QRect(option.rect)
        offset = 3
        psize = 7
        prect.setX(prect.x() + 3)
        prect.setY(prect.bottom() - psize - offset)
        prect.setWidth(prect.width() - offset - 1)
        prect.setHeight(psize)
        painter.fillRect(prect, QtGui.QColor(220, 220, 220))
        painter.drawRect(prect)


        prect = QtCore.QRect(prect)
        prect.setX(prect.x() + 2)
        prect.setY(prect.y() + 2)
        prect.setWidth(prect.width() - 1)
        prect.setHeight(prect.height() - 1)

        if data.isRunning():
            cprect = QtCore.QRect(prect)
            cprect.setWidth(cprect.width() * 0.75)
            painter.fillRect(cprect, self.running.dark(120))

        gradient = QtGui.QLinearGradient(0, prect.top(), 0, prect.bottom())
        gradient.setColorAt(0, QtGui.QColor(255, 255, 255, 0))
        gradient.setColorAt(0.25, QtCore.Qt.white)
        gradient.setColorAt(0.50, QtGui.QColor(255, 255, 255, 0))
        gradient.setColorAt(1, QtGui.QColor(255, 255, 255, 0))

        painter.fillRect(prect, gradient)


        if option.state & QtGui.QStyle.State_Selected:
            painter.fillRect(option.rect, QtGui.QColor(128, 128, 128, 128))

        painter.restore()

    def sizeHint(self, option, index):
        return self.size


        


class Simulation:
    job_status_type_reverse = {"JOB_QUEUE_NOT_ACTIVE" : 0,
                               "JOB_QUEUE_LOADING" : 1,
                               "JOB_QUEUE_NULL" : 2,
                               "JOB_QUEUE_WAITING" : 3,
                               "JOB_QUEUE_PENDING" : 4,
                               "JOB_QUEUE_RUNNING" : 5,
                               "JOB_QUEUE_DONE" : 6,
                               "JOB_QUEUE_EXIT" : 7,
                               "JOB_QUEUE_RUN_OK" : 8,
                               "JOB_QUEUE_RUN_FAIL" : 9,
                               "JOB_QUEUE_ALL_OK" : 10,
                               "JOB_QUEUE_ALL_FAIL" : 11,
                               "JOB_QUEUE_USER_KILLED" : 12,
                               "JOB_QUEUE_MAX_STATE" : 13}

    job_status_type = {0 : "JOB_QUEUE_NOT_ACTIVE",
                       1 : "JOB_QUEUE_LOADING",
                       2 : "JOB_QUEUE_NULL",
                       3 : "JOB_QUEUE_WAITING",
                       4 : "JOB_QUEUE_PENDING",
                       5 : "JOB_QUEUE_RUNNING",
                       6 : "JOB_QUEUE_DONE",
                       7 : "JOB_QUEUE_EXIT",
                       8 : "JOB_QUEUE_RUN_OK",
                       9 : "JOB_QUEUE_RUN_FAIL",
                       10 : "JOB_QUEUE_ALL_OK",
                       11 : "JOB_QUEUE_ALL_FAIL",
                       12 : "JOB_QUEUE_USER_KILLED",
                       13 : "JOB_QUEUE_MAX_STATE"}


    def __init__(self, name):
        self.name = name
        self.status = 0 #JOB_QUEUE_NOT_ACTIVE
        self.statuslog = []

    def checkStatus(self, type):
        return self.status == self.job_status_type_reverse[type]

    def isWaiting(self):
        return self.checkStatus("JOB_QUEUE_WAITING") or self.checkStatus("JOB_QUEUE_PENDING")

    def isRunning(self):
        return self.checkStatus("JOB_QUEUE_RUNNING")

    def hasFailed(self):
        return self.checkStatus("JOB_QUEUE_ALL_FAIL")

    def notActive(self):
        return self.checkStatus("JOB_QUEUE_NOT_ACTIVE")

    def finishedSuccesfully(self):
        return self.checkStatus("JOB_QUEUE_ALL_OK")

    def setStatus(self, status):
        if len(self.statuslog) == 0 or not self.statuslog[len(self.statuslog) - 1] == status:
            self.statuslog.append(status)

        self.status = status


class Cogwheel(QtGui.QWidget):

    def __init__(self, color=QtGui.QColor(128, 128, 128), size=64, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.size = size
        qsize = QtCore.QSize(size, size)
        self.setMaximumSize(qsize)
        self.setMinimumSize(qsize)

        self.color = color
        self.inc = 0
        self.step = 2.5

        self.createCogwheel(size)

        timer = QtCore.QTimer(self)
        self.connect(timer, QtCore.SIGNAL("timeout()"), self, QtCore.SLOT("update()"))
        timer.start(16)

        self.running = False


    def paintEvent(self, paintevent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.contentsRect()

        painter.setPen(QtGui.QColor(255, 255, 255, 0))

        painter.setClipRect(rect)
        painter.translate(rect.center())
        painter.rotate(self.step * self.inc)
        self.drawCogwheel(painter)

        r = (self.size / 2.0) * 0.4
        painter.setBrush(QtGui.QBrush(self.color.light(150)))
        painter.drawEllipse(QtCore.QPointF(0, 0), r, r)

        if self.running:
            self.inc += 1


    def drawCogwheel(self, painter):
        painter.save()
        painter.setBrush(QtGui.QBrush(self.color))
        painter.drawPolygon(QtGui.QPolygonF(self.points), len(self.points))
        painter.restore()


    def createCogwheel(self, size):
        self.points = []
        r1 = (size / 2.0) - 1.0
        r2 = 0.80
        teeth = 9
        out = False
        for t in frange(0.0, 2 * math.pi, 2 * math.pi / (teeth * 2.0)):
            x = r1 * math.cos(t)
            y = r1 * math.sin(t)
            if out:
                self.points.append(QtCore.QPointF(x, y))
                self.points.append(QtCore.QPointF(r2 * x, r2 * y))
            else:
                self.points.append(QtCore.QPointF(r2 * x, r2 * y))
                self.points.append(QtCore.QPointF(x, y))
            out = not out

    def setRunning(self, bool):
        self.running = bool

class LegendMarker(QtGui.QWidget):
    def __init__(self, color, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.setMaximumSize(QtCore.QSize(12, 12))
        self.setMinimumSize(QtCore.QSize(12, 12))

        self.color = color

    def paintEvent(self, paintevent):
        painter = QtGui.QPainter(self)

        rect = self.contentsRect()

        rect.setWidth(rect.width() - 1)
        rect.setHeight(rect.height() - 1)
        painter.drawRect(rect)

        rect.setX(rect.x() + 1)
        rect.setY(rect.y() + 1)

        painter.fillRect(rect, self.color)

class Legend(QtGui.QHBoxLayout):
    def __init__(self, legend, color, parent=None):
        QtGui.QHBoxLayout.__init__(self, parent)

        self.addWidget(LegendMarker(color, parent))
        self.addWidget(QtGui.QLabel(legend))


class RunWidget(HelpedWidget):
    run_mode_type = {"ENKF_ASSIMILATION" : 1, "ENSEMBLE_EXPERIMENT" : 2, "ENSEMBLE_PREDICTION" : 3, "INIT_ONLY" : 4}
    state_enum = {"UNDEFINED" : 0, "SERIALIZED" : 1, "FORECAST" : 2, "ANALYZED" : 4, "BOTH" : 6}

    def __init__(self, parent=None):
        HelpedWidget.__init__(self, parent, widgetLabel="", helpLabel="widget_run")

        self.membersList = QtGui.QListWidget(self)
        self.membersList.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)

        self.membersList.setViewMode(QtGui.QListView.IconMode)
        self.membersList.setMovement(QtGui.QListView.Static)
        self.membersList.setResizeMode(QtGui.QListView.Adjust)
        self.membersList.setGridSize(QtCore.QSize(32, 16))
        self.membersList.setSelectionRectVisible(False)

        memberLayout = QtGui.QFormLayout()
        memberLayout.setLabelAlignment(QtCore.Qt.AlignRight)

        self.runpathLabel = QtGui.QLabel("")
        font = self.runpathLabel.font()
        font.setWeight(QtGui.QFont.Bold)
        self.runpathLabel.setFont(font)

        memberLayout.addRow("Runpath:", self.runpathLabel)

        membersCheckPanel = ListCheckPanel(self.membersList)
        #membersCheckPanel.insertWidget(0, QtGui.QLabel("Members"))

        self.simulateFrom = ValidatedTimestepCombo(parent, fromLabel="Start", toLabel="End of history")
        self.simulateTo = ValidatedTimestepCombo(parent, fromLabel="End of history", toLabel="End of prediction")

        self.startState = QtGui.QComboBox(self)
        self.startState.setMaximumWidth(100)
        self.startState.setToolTip("Select state")
        self.startState.addItem("Analyzed")
        self.startState.addItem("Forecast")

        startLayout = QtGui.QHBoxLayout()
        startLayout.addWidget(self.simulateFrom)
        startLayout.addWidget(self.startState)

        memberLayout.addRow("Run simulation from: ", startLayout)
        memberLayout.addRow("Run simulation to: ", self.simulateTo)
        memberLayout.addRow("Mode: ", self.createRadioButtons())
        memberLayout.addRow(membersCheckPanel)
        memberLayout.addRow("Members:", self.membersList)

        self.actionButton = QtGui.QPushButton("Run simulation")

        self.connect(self.actionButton, QtCore.SIGNAL('clicked()'), self.run)

        actionLayout = QtGui.QHBoxLayout()
        actionLayout.addStretch(1)
        actionLayout.addWidget(self.actionButton)
        actionLayout.addStretch(1)

        memberLayout.addRow(createSpace(10))
        memberLayout.addRow(actionLayout)

        memberLayout.addRow(widgets.util.createSeparator())

        self.simulationProgress = QtGui.QProgressBar()
        self.simulationProgress.setValue(0)
        self.connect(self.simulationProgress, QtCore.SIGNAL('setValue(int)'), self.updateProgress)

        self.cogwheel = Cogwheel(size=20)

        progressLayout = QtGui.QHBoxLayout()
        progressLayout.addWidget(self.simulationProgress)
        progressLayout.addWidget(self.cogwheel)
        memberLayout.addRow(progressLayout)


        self.simulationList = SimulationList()
        self.simulationList.setSortingEnabled(True)

        memberLayout.addRow(self.simulationList)

        legendLayout = QtGui.QHBoxLayout()
        legendLayout.addLayout(Legend("Not active", SimulationItemDelegate.notactive))
        legendLayout.addLayout(Legend("Waiting/Pending", SimulationItemDelegate.waiting))
        legendLayout.addLayout(Legend("Running", SimulationItemDelegate.running))
        legendLayout.addLayout(Legend("Loading/etc.", SimulationItemDelegate.unknown))
        legendLayout.addLayout(Legend("Failed", SimulationItemDelegate.failed))
        legendLayout.addLayout(Legend("Finished", SimulationItemDelegate.finished))
        memberLayout.addRow(legendLayout)

        self.addLayout(memberLayout)

        self.setRunpath("...")

        self.modelConnect("ensembleResized()", self.fetchContent)
        self.modelConnect("runpathChanged()", self.fetchContent)

        self.rbAssimilation.toggle()


    def run(self):
        ert = self.getModel()
        selectedMembers = getItemsFromList(self.membersList)

        selectedMembers = [int(member) for member in selectedMembers]

        if len(selectedMembers) == 0:
            QtGui.QMessageBox.warning(self, "Missing data", "At least one member must be selected!")
            return

        simFrom = self.simulateFrom.getSelectedValue()
        simTo = self.simulateTo.getSelectedValue()

        if self.rbAssimilation.isChecked():
            mode = self.run_mode_type["ENKF_ASSIMILATION"]
        else:
            if simTo == -1: # -1 == End
                mode = self.run_mode_type["ENSEMBLE_PREDICTION"]
            else:
                mode = self.run_mode_type["ENSEMBLE_EXPERIMENT"]

        state = self.state_enum["ANALYZED"]
        if self.startState.currentText() == "Forecast" and not simFrom == 0:
            state = self.state_enum["FORECAST"]

        init_step_parameter = simFrom

        #if mode == run_mode_type["ENKF_ASSIMILATION"]:
        #    init_step_parameter = simFrom
        #elif mode == run_mode_type["ENSEMBLE_EXPERIMENT"]:
        #    init_step_parameter = 0
        #else:
        #    init_step_parameter = self.historyLength

        simulations = {}
        for member in selectedMembers:
            simulations[member] = SimulationItem(Simulation(member))
            self.simulationList.addItem(simulations[member])


        self.runthread = threading.Thread(name="enkf_main_run")
        def action():
            self.setGUIEnabled(False)
            boolVector = ert.createBoolVector(self.membersList.count(), selectedMembers)
            boolPtr = ert.getBoolVectorPtr(boolVector)

            ert.enkf.enkf_main_run(ert.main, mode, boolPtr, init_step_parameter, simFrom, state)
            ert.freeBoolVector(boolVector)
            self.setGUIEnabled(True)

        self.runthread.setDaemon(True)
        self.runthread.run = action                        

        self.pollthread = threading.Thread(name="polling_thread")
        def poll():
            while not ert.enkf.site_config_queue_is_running(ert.site_config):
                time.sleep(0.5)

            while(self.runthread.isAlive()):
                for member in selectedMembers:
                    state = ert.enkf.enkf_main_iget_state(ert.main, member)
                    status = ert.enkf.enkf_state_get_run_status(state)

                    if not status == Simulation.job_status_type_reverse["JOB_QUEUE_NOT_ACTIVE"]:
                        #start_time = ert.enkf.enkf_state_get_start_time(state)
                        #print time.ctime(start_time), start_time
                        pass

                    simulations[member].simulation.setStatus(status)
                    simulations[member].updateSimulation()

                totalCount = len(simulations.keys())
                succesCount = 0
                for key in simulations.keys():
                    if simulations[key].simulation.finishedSuccesfully():
                        succesCount+=1

                count = (100 * succesCount / totalCount)
                self.simulationProgress.emit(QtCore.SIGNAL("setValue(int)"), count)

                time.sleep(0.5)

        self.pollthread.setDaemon(True)
        self.pollthread.run = poll

        self.runthread.start()
        self.pollthread.start()


    def updateProgress(self, value):
        self.simulationProgress.setValue(value)

    def setRunpath(self, runpath):
        #self.runpathLabel.setText("Runpath: " + runpath)
        self.runpathLabel.setText(runpath)

    def fetchContent(self):
        data = self.getFromModel()

        self.historyLength = data["history_length"]

        self.membersList.clear()

        for member in data["members"]:
            self.membersList.addItem("%03d" % (member))
        #self.membersList.addItem(str(member))

        self.setRunpath(data["runpath"])

        self.simulateFrom.setHistoryLength(self.historyLength)
        self.simulateTo.setFromValue(self.historyLength)
        self.simulateTo.setToValue(-1)
        self.simulateTo.setMinTimeStep(0)
        self.simulateTo.setMaxTimeStep(self.historyLength)

        self.membersList.selectAll()


    def initialize(self, ert):
        ert.setTypes("enkf_main_get_ensemble_size", ertwrapper.c_int)
        ert.setTypes("enkf_main_get_history_length", ertwrapper.c_int)
        ert.setTypes("model_config_get_runpath_as_char", ertwrapper.c_char_p)

        ert.setTypes("enkf_main_iget_state", ertwrapper.c_long, ertwrapper.c_int)
        ert.setTypes("enkf_state_get_run_status", ertwrapper.c_int)
        ert.setTypes("site_config_queue_is_running")
        ert.setTypes("enkf_state_get_start_time")


    def getter(self, ert):
        members = ert.enkf.enkf_main_get_ensemble_size(ert.main)
        historyLength = ert.enkf.enkf_main_get_history_length(ert.main)
        runpath = ert.enkf.model_config_get_runpath_as_char(ert.model_config)

        return {"members" : range(members), "history_length" : historyLength, "runpath" : runpath}


    def rbToggle(self):
        if self.rbAssimilation.isChecked():
            self.membersList.setSelectionEnabled(False)
            self.membersList.selectAll()
        else:
            self.membersList.setSelectionEnabled(True)

    def createRadioButtons(self):
        radioLayout = QtGui.QVBoxLayout()
        radioLayout.setSpacing(2)
        self.rbAssimilation = QtGui.QRadioButton("EnKF assimilation")
        radioLayout.addWidget(self.rbAssimilation)
        self.rbExperiment = QtGui.QRadioButton("Ensemble experiment")
        radioLayout.addWidget(self.rbExperiment)

        self.connect(self.rbAssimilation, QtCore.SIGNAL('toggled(bool)'), lambda : self.rbToggle())
        self.connect(self.rbExperiment, QtCore.SIGNAL('toggled(bool)'), lambda : self.rbToggle())

        return radioLayout

    def setGUIEnabled(self, state):
        if not self.rbAssimilation.isChecked():
            self.membersList.setSelectionEnabled(state)

        self.rbAssimilation.setEnabled(state)
        self.rbExperiment.setEnabled(state)
        self.simulateFrom.setEnabled(state)
        self.simulateTo.setEnabled(state)
        self.startState.setEnabled(state)
        self.actionButton.setEnabled(state)
        
        self.cogwheel.setRunning(not state)


class RunPanel(QtGui.QFrame):
    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.Panel)
        self.setFrameShadow(QtGui.QFrame.Raised)

        panelLayout = QtGui.QVBoxLayout()
        self.setLayout(panelLayout)

        #        button = QtGui.QPushButton("Refetch")
        #        self.connect(button, QtCore.SIGNAL('clicked()'), ContentModel.updateObservers)
        #        panelLayout.addWidget(button)

        panelLayout.addWidget(RunWidget())



        




