from PyQt4 import QtGui, QtCore
from widgets.cogwheel import Cogwheel
from pages.run.legend import Legend
from pages.run.simulation import SimulationItemDelegate, SimulationList, SimulationItem, Simulation, SimulationPanel, SimulationStatistics

import threading
import time
import ertwrapper
from widgets.util import getItemsFromList

class JobsDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setModal(True)
        self.setWindowTitle("Running jobs")
        self.setMinimumWidth(250)
        #self.setMinimumHeight(250)

        self.ctrl = JobsDialogController(self) 

        self.simulationProgress = QtGui.QProgressBar()
        self.simulationProgress.setValue(0)
        self.connect(self.simulationProgress, QtCore.SIGNAL('setValue(int)'), self.updateProgress)

        self.cogwheel = Cogwheel(size=20)

        memberLayout = QtGui.QVBoxLayout()

        progressLayout = QtGui.QHBoxLayout()
        progressLayout.addWidget(self.simulationProgress)
        progressLayout.addWidget(self.cogwheel)
        memberLayout.addLayout(progressLayout)


        simulationLayout = QtGui.QHBoxLayout()
        self.simulationList = SimulationList()
        self.simulationList.contextMenuEvent = self.contextMenu
        self.connect(self.simulationList, QtCore.SIGNAL('itemSelectionChanged()'), self.ctrl.selectSimulation)
        simulationLayout.addWidget(self.simulationList)
        self.simulationPanel = SimulationPanel()
        simulationLayout.addWidget(self.simulationPanel)
        memberLayout.addLayout(simulationLayout)

        legendLayout = QtGui.QHBoxLayout()
        legendLayout.addLayout(Legend("Not active", SimulationItemDelegate.notactive))
        legendLayout.addLayout(Legend("Waiting/Pending", SimulationItemDelegate.waiting))
        legendLayout.addLayout(Legend("Running", SimulationItemDelegate.running))
        legendLayout.addLayout(Legend("Loading/etc.", SimulationItemDelegate.unknown))
        legendLayout.addLayout(Legend("User killed", SimulationItemDelegate.userkilled))
        legendLayout.addLayout(Legend("Failed", SimulationItemDelegate.failed))
        legendLayout.addLayout(Legend("Finished", SimulationItemDelegate.finished))
        memberLayout.addLayout(legendLayout)


        self.doneButton = QtGui.QPushButton("Done", self)
        self.connect(self.doneButton, QtCore.SIGNAL('clicked()'), self.accept)

        buttonLayout = QtGui.QHBoxLayout()

        self.estimateLabel = QtGui.QLabel()
        buttonLayout.addWidget(self.estimateLabel)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.doneButton)

        memberLayout.addSpacing(10)
        memberLayout.addLayout(buttonLayout)

        self.setLayout(memberLayout)


    def createAction(self, name, func, parent=None):
        action = QtGui.QAction(name, parent)
        action.connect(action, QtCore.SIGNAL("triggered()"), func)
        return action

    def contextMenu(self, event):
        menu = QtGui.QMenu(self.simulationList)
        selectAll = self.createAction("Select all", self.simulationList.selectAll)
        unselectAll = self.createAction("Unselect all", self.simulationList.clearSelection)
        selectRunning = self.createAction("Select all running", lambda : self.ctrl.select(Simulation.RUNNING))
        selectFailed = self.createAction("Select all failed", lambda : self.ctrl.select(Simulation.ALL_FAIL))
        selectUserKilled = self.createAction("Select all user killed", lambda : self.ctrl.select(Simulation.USER_KILLED))
        selectWaiting = self.createAction("Select all waiting", lambda : self.ctrl.select(Simulation.WAITING, Simulation.PENDING))

        menu.addAction(selectAll)
        menu.addAction(unselectAll)
        menu.addAction(selectWaiting)
        menu.addAction(selectRunning)
        menu.addAction(selectFailed)
        menu.addAction(selectUserKilled)
        menu.exec_(event.globalPos())
        

    def closeEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if not event.key() == QtCore.Qt.Key_Escape:
            QtGui.QDialog.keyPressEvent(self, event)

    def updateProgress(self, value):
        self.simulationProgress.setValue(value)

    def setRunningState(self, state):
        self.cogwheel.setRunning(state)
        self.doneButton.setEnabled(not state)

    def start(self, **kwargs):
        self.open()

        self.ctrl.start(**kwargs)

        self.exec_()



class JobsDialogController:
    def __init__(self, view):
        self.view = view
        self.initialized = False

    def select(self, *states):
        self.view.simulationList.clearSelection()

        items = getItemsFromList(self.view.simulationList, lambda item : item, selected=False)

        for state in states:
            for item in items:
                if item.simulation.checkStatus(state):
                    item.setSelected(True)

    def selectSimulation(self):
        selection = getItemsFromList(self.view.simulationList, lambda item : item.simulation)
        self.view.simulationPanel.setSimulations(selection)


    def initialize(self, ert):
        if not self.initialized:
            ert.setTypes("enkf_main_iget_state", ertwrapper.c_long, ertwrapper.c_int)
            ert.setTypes("enkf_state_get_run_status", ertwrapper.c_int)
            ert.setTypes("site_config_queue_is_running")
            ert.setTypes("enkf_state_get_start_time")
            ert.setTypes("enkf_state_get_submit_time")
            self.initialized = True

    def start(self, **kwargs):
        ert = kwargs["ert"]
        memberCount = kwargs["memberCount"]
        selectedMembers = kwargs["selectedMembers"]
        mode = kwargs["mode"]
        init_step_parameter = kwargs["init_step_parameter"]
        simFrom = kwargs["simFrom"]
        simTo = kwargs["simTo"]
        state = kwargs["state"]


        self.initialize(ert)
        self.view.simulationPanel.setModel(ert)

        self.statistics = SimulationStatistics()
        simulations = {}
        for member in selectedMembers:
            simulations[member] = SimulationItem(Simulation(member, self.statistics))
            self.view.simulationList.addItem(simulations[member])


        self.runthread = threading.Thread(name="enkf_main_run")
        def run():
            self.view.setRunningState(True)
            boolVector = ert.createBoolVector(memberCount, selectedMembers)
            boolPtr = ert.getBoolVectorPtr(boolVector)

            ert.enkf.enkf_main_run(ert.main, mode, boolPtr, init_step_parameter, simFrom, state)
            ert.freeBoolVector(boolVector)
            self.view.setRunningState(False)

        self.runthread.setDaemon(True)
        self.runthread.run = run


        self.pollthread = threading.Thread(name="polling_thread")
        def poll():
            while not ert.enkf.site_config_queue_is_running(ert.site_config):
                time.sleep(0.5)

            while(self.runthread.isAlive()):
                for member in selectedMembers:
                    state = ert.enkf.enkf_main_iget_state(ert.main, member)
                    status = ert.enkf.enkf_state_get_run_status(state)

                    simulations[member].simulation.setStatus(status)

                    if not status == Simulation.NOT_ACTIVE:
                        start_time = ert.enkf.enkf_state_get_start_time(state)
                        submit_time = ert.enkf.enkf_state_get_submit_time(state)

                        simulations[member].simulation.setStartTime(start_time)
                        simulations[member].simulation.setSubmitTime(submit_time)


                    simulations[member].updateSimulation()

                totalCount = len(simulations.keys())
                succesCount = 0
                for key in simulations.keys():
                    if simulations[key].simulation.finishedSuccesfully():
                        succesCount+=1

                count = (100 * succesCount / totalCount)
                self.view.simulationProgress.emit(QtCore.SIGNAL("setValue(int)"), count)
                self.view.simulationPanel.emit(QtCore.SIGNAL("simulationsUpdated()"))

                if self.statistics.jobsPerSecond() > 0:
                    self.view.estimateLabel.setText("Estimated finished in %.2f secs" % (self.statistics.estimate(len(simulations))))
                else:
                    self.view.estimateLabel.setText("")
                time.sleep(0.5)

        self.pollthread.setDaemon(True)
        self.pollthread.run = poll

        self.statistics.startTiming()
        self.runthread.start()
        self.pollthread.start()
