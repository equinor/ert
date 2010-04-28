from __future__ import division
from PyQt4 import QtGui, QtCore
from widgets.util import resourceIcon, resourceStateIcon, shortTime
import time
import ertwrapper


class SimulationList(QtGui.QListWidget):
    def __init__(self):
        QtGui.QListWidget.__init__(self)

        self.setViewMode(QtGui.QListView.IconMode)
        self.setMovement(QtGui.QListView.Static)
        self.setResizeMode(QtGui.QListView.Adjust)

        self.setItemDelegate(SimulationItemDelegate())
        self.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.setSelectionRectVisible(False)

        self.setSortingEnabled(True)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        
class SimulationItem(QtGui.QListWidgetItem):
    def __init__(self, simulation):
        self.simulation = simulation
        QtGui.QListWidgetItem.__init__(self, type=9901)
        self.setData(QtCore.Qt.DisplayRole, self.simulation)

    def __ge__(self, other):
        return self.simulation >= other.simulation

    def __lt__(self, other):
        return not self >= other


class SimulationItemDelegate(QtGui.QStyledItemDelegate):
    waiting = QtGui.QColor(164, 164, 255)
    running = QtGui.QColor(200, 255, 200)
    failed = QtGui.QColor(255, 200, 200)
    unknown = QtGui.QColor(255, 200, 128)
    userkilled = QtGui.QColor(255, 255, 200)
    finished = QtGui.QColor(200, 200, 200)
    notactive = QtGui.QColor(255, 255, 255)

    size = QtCore.QSize(32, 18)

    def __init__(self):
        QtGui.QStyledItemDelegate.__init__(self)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        #data = index.model().data(index)
        data = index.data(QtCore.Qt.DisplayRole)
        #data = None

        if data is None:
            data = Simulation("0")
            data.status = 0
        else:
            data = data.toPyObject()

        if data.isWaiting():
            color = self.waiting
        elif data.isRunning():
            color = self.running
        elif data.finishedSuccessfully():
            color = self.finished
        elif data.hasFailed():
            color = self.failed
        elif data.notActive():
            color = self.notactive
        elif data.isUserKilled():
            color = self.userkilled
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

        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter.drawRect(rect)

        if option.state & QtGui.QStyle.State_Selected:
            painter.fillRect(option.rect, QtGui.QColor(255, 255, 255, 150))

        painter.drawText(rect, QtCore.Qt.AlignCenter + QtCore.Qt.AlignVCenter, str(data.name))

        painter.restore()

    def sizeHint(self, option, index):
        return self.size


class SimulationPanel(QtGui.QStackedWidget):

    def __init__(self, parent=None):
        QtGui.QStackedWidget.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.Panel)
        self.setFrameShadow(QtGui.QFrame.Raised)

        self.setMinimumWidth(200)
        self.setMaximumWidth(200)

        self.ctrl = SimulationPanelController(self)

        self.createNoSelectionsPanel()
        self.createSingleSelectionsPanel()
        self.createManySelectionsPanel()

        self.addWidget(self.noSimulationsPanel)
        self.addWidget(self.singleSimulationsPanel)
        self.addWidget(self.manySimulationsPanel)

        
    def createButtons(self):
        self.killButton = QtGui.QToolButton(self)
        self.killButton.setIcon(resourceIcon("cross"))
        self.killButton.setToolTip("Kill job")
        self.connect(self.killButton, QtCore.SIGNAL('clicked()'), self.ctrl.kill)

        self.restartButton = QtGui.QToolButton(self)
        self.restartButton.setIcon(resourceIcon("refresh"))
        self.restartButton.setToolTip("Restart job")
        self.connect(self.restartButton, QtCore.SIGNAL('clicked()'), lambda : self.ctrl.restart(False))

        self.rrButton = QtGui.QToolButton(self)
        self.rrButton.setIcon(resourceIcon("refresh_resample"))
        self.rrButton.setToolTip("Resample and restart job")
        self.connect(self.rrButton, QtCore.SIGNAL('clicked()'), lambda : self.ctrl.restart(True))

        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addWidget(self.killButton)
        buttonLayout.addWidget(self.restartButton)
        buttonLayout.addWidget(self.rrButton)

        return buttonLayout

    def createButtonedLayout(self, layout, stretch=True):
        btnlayout = QtGui.QVBoxLayout()
        btnlayout.addLayout(layout)

        if stretch:
            btnlayout.addStretch(1)

        btnlayout.addLayout(self.createButtons())
        return btnlayout


    def createManySelectionsPanel(self):
        self.manySimulationsPanel = QtGui.QWidget()

        layout = QtGui.QVBoxLayout()
        label = QtGui.QLabel("Selected jobs:")
        label.setAlignment(QtCore.Qt.AlignHCenter)
        layout.addWidget(label)

        self.selectedSimulationsLabel = QtGui.QLabel()
        self.selectedSimulationsLabel.setWordWrap(True)
        font = self.selectedSimulationsLabel.font()
        font.setWeight(QtGui.QFont.Bold)
        self.selectedSimulationsLabel.setFont(font)

        scrolledLabel = QtGui.QScrollArea()
        scrolledLabel.setWidget(self.selectedSimulationsLabel)
        scrolledLabel.setWidgetResizable(True)
        layout.addWidget(scrolledLabel)

        self.manySimulationsPanel.setLayout(self.createButtonedLayout(layout, False))

    def createSingleSelectionsPanel(self):
        self.singleSimulationsPanel = QtGui.QWidget()

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)
        self.jobLabel = QtGui.QLabel()
        self.submitLabel = QtGui.QLabel()
        self.startLabel = QtGui.QLabel()
        self.finishLabel = QtGui.QLabel()
        self.waitingLabel = QtGui.QLabel()
        self.runningLabel = QtGui.QLabel()
        self.stateLabel = QtGui.QLabel()

        layout.addRow("Job #:", self.jobLabel)
        layout.addRow("Submitted:", self.submitLabel)
        layout.addRow("Started:", self.startLabel)
        layout.addRow("Finished:", self.finishLabel)
        layout.addRow("Waiting:", self.runningLabel)
        layout.addRow("Running:", self.waitingLabel)
        layout.addRow("State:", self.stateLabel)

        self.singleSimulationsPanel.setLayout(self.createButtonedLayout(layout))


    def createNoSelectionsPanel(self):
        self.noSimulationsPanel = QtGui.QWidget()

        layout = QtGui.QVBoxLayout()
        label = QtGui.QLabel("Pause queue after currently running jobs are finished:")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.pauseButton = QtGui.QToolButton(self)
        self.pauseButton.setIcon(resourceIcon("pause"))
        self.pauseButton.setCheckable(True)
        self.connect(self.pauseButton, QtCore.SIGNAL('clicked()'), lambda : self.ctrl.pause(self.pauseButton.isChecked()))


        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.pauseButton)
        buttonLayout.addStretch(1)
        layout.addLayout(buttonLayout)

        label = QtGui.QLabel("Remove all jobs from the queue:")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.killAllButton = QtGui.QToolButton(self)
        self.killAllButton.setIcon(resourceIcon("cancel"))
        self.connect(self.killAllButton, QtCore.SIGNAL('clicked()'), self.ctrl.killAll)

        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.killAllButton)
        buttonLayout.addStretch(1)

        layout.addLayout(buttonLayout)

        self.noSimulationsPanel.setLayout(layout)



    def setSimulations(self, selection=None):
        if selection is None: selection = []
        self.ctrl.setSimulations(selection)

#    def markText(self, a, b):
#        if b.isRunning():
#            c = SimulationItemDelegate.running
#        elif b.isWaiting():
#            c = SimulationItemDelegate.waiting
#        else:
#            c = QtGui.QColor(255, 255, 255, 0)
#
#        color = "rgb(%d, %d, %d)" % (c.red(), c.green(), c.blue())
#
#        b = "<span style='background: " + color + ";'>" + str(b) + "</span>"
#
#        if not a == "":
#            return a + " " + b
#        else:
#            return b

    def setModel(self, ert):
        self.ctrl.setModel(ert)

    def setSimulationStatistics(self, statistics):
        self.ctrl.setSimulationStatistics(statistics)


class SimulationPanelController:
    def __init__(self, view):
        self.view = view
        self.initialized = False
        self.selectedSimulations = []
        self.view.connect(self.view, QtCore.SIGNAL('simulationsUpdated()'), self.showSelectedSimulations)

    def initialize(self, ert):
        if not self.initialized:
            ert.setTypes("job_queue_get_pause", library = ert.job_queue)
            ert.setTypes("job_queue_set_pause_on", library = ert.job_queue)
            ert.setTypes("job_queue_set_pause_off", library = ert.job_queue)
            ert.setTypes("job_queue_user_exit", library = ert.job_queue)
            ert.setTypes("enkf_main_iget_state", argtypes=ertwrapper.c_int)
            ert.setTypes("enkf_state_kill_simulation", None)
            ert.setTypes("enkf_state_resubmit_simulation", None, ertwrapper.c_int)
            ert.setTypes("enkf_state_get_run_status", ertwrapper.c_int)
            ert.setTypes("site_config_get_job_queue")
            self.initialized = True

    def setModel(self, ert):
        self.initialize(ert)
        self.ert = ert

    def kill(self):
        """Kills the selected simulations."""
        for simulation in self.selectedSimulations:
            state = self.ert.enkf.enkf_main_iget_state(self.ert.main, simulation.name)
            status = self.ert.enkf.enkf_state_get_run_status(state)

            #if status == Simulation.RUNNING:
            self.ert.enkf.enkf_state_kill_simulation(state)

    def restart(self, resample):
        """Restarts the selected simulations. May also resample."""
        for simulation in self.selectedSimulations:
            state = self.ert.enkf.enkf_main_iget_state(self.ert.main, simulation.name)
            status = self.ert.enkf.enkf_state_get_run_status(state)

            #if status == Simulation.USER_KILLED:
            self.ert.enkf.enkf_state_resubmit_simulation(state , resample)

    def pause(self, pause):
        """Pause the job queue after the currently running jobs are finished."""
        job_queue = self.ert.enkf.site_config_get_job_queue(self.ert.site_config)

        if pause:
            self.statistics.stop()
            self.ert.job_queue.job_queue_set_pause_on(job_queue)
        else:
            self.statistics.startTiming()
            self.ert.job_queue.job_queue_set_pause_off(job_queue)

    def killAll(self):
        killAll = QtGui.QMessageBox.question(self.view, "Remove all jobs?", "Are you sure you want to remove all jobs from the queue?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

        if killAll == QtGui.QMessageBox.Yes:
            job_queue = self.ert.enkf.site_config_get_job_queue(self.ert.site_config)
            self.ert.job_queue.job_queue_user_exit(job_queue)

    def showSelectedSimulations(self):
        if len(self.selectedSimulations) >= 2:
            members = reduce(lambda a, b: str(a) + " " + str(b), sorted(self.selectedSimulations))
            self.view.selectedSimulationsLabel.setText(members)
        elif len(self.selectedSimulations) == 1:
            sim = self.selectedSimulations[0]
            self.view.jobLabel.setText(str(sim.name))
            self.view.submitLabel.setText(shortTime(sim.submitTime))
            self.view.startLabel.setText(shortTime(sim.startTime))
            self.view.finishLabel.setText(shortTime(sim.finishedTime))

            if sim.startTime == -1:
                runningTime = "-"
            elif sim.finishedTime > -1:
                runningTime = sim.finishedTime - sim.startTime
            else:
                runningTime = int(time.time()) - sim.startTime


            if sim.submitTime == -1:
                waitingTime = "-"
            elif sim.startTime > -1:
                waitingTime = sim.startTime - sim.submitTime
            else:
                waitingTime = int(time.time()) - sim.submitTime

            self.view.runningLabel.setText(str(waitingTime) + " secs")
            self.view.waitingLabel.setText(str(runningTime) + " secs")

            status = Simulation.job_status_type[sim.status]
            status = status[10:]
            self.view.stateLabel.setText(status)


    def setSimulations(self, selection=None):
        if selection is None: selection = []
        self.selectedSimulations = selection

        if len(selection) >= 2:
            self.view.setCurrentWidget(self.view.manySimulationsPanel)
        elif len(selection) == 1:
            self.view.setCurrentWidget(self.view.singleSimulationsPanel)
        else:
            self.view.setCurrentWidget(self.view.noSimulationsPanel)

        self.showSelectedSimulations()

    def setSimulationStatistics(self, statistics):
        self.statistics = statistics


class Simulation:
    # These "enum" values are all copies from the header file "basic_queue_driver.h".

    NOT_ACTIVE  =    1
    LOADING     =    2
    WAITING     =    4
    PENDING     =    8
    RUNNING     =   16
    DONE        =   32
    EXIT        =   64
    RUN_OK      =  128
    RUN_FAIL    =  256
    ALL_OK      =  512
    ALL_FAIL    = 1024
    USER_KILLED = 2048
    USER_EXIT   = 4096

#
# Observe that the status strings are available from the function: libjob_queue.job_queue_status_name( status_code )
#


    job_status_type_reverse = {"JOB_QUEUE_NOT_ACTIVE" : NOT_ACTIVE,
                               "JOB_QUEUE_LOADING" : LOADING,
                               "JOB_QUEUE_WAITING" : WAITING,
                               "JOB_QUEUE_PENDING" : PENDING,
                               "JOB_QUEUE_RUNNING" : RUNNING,
                               "JOB_QUEUE_DONE" : DONE,
                               "JOB_QUEUE_EXIT" : EXIT,
                               "JOB_QUEUE_RUN_OK" : RUN_OK,
                               "JOB_QUEUE_RUN_FAIL" : RUN_FAIL,
                               "JOB_QUEUE_ALL_OK" : ALL_OK,
                               "JOB_QUEUE_ALL_FAIL" : ALL_FAIL,
                               "JOB_QUEUE_USER_KILLED" : USER_KILLED}

    job_status_type = {NOT_ACTIVE : "JOB_QUEUE_NOT_ACTIVE",
                       LOADING : "JOB_QUEUE_LOADING",
                       WAITING : "JOB_QUEUE_WAITING",
                       PENDING : "JOB_QUEUE_PENDING",
                       RUNNING : "JOB_QUEUE_RUNNING",
                       DONE : "JOB_QUEUE_DONE",
                       EXIT : "JOB_QUEUE_EXIT",
                       RUN_OK : "JOB_QUEUE_RUN_OK",
                       RUN_FAIL : "JOB_QUEUE_RUN_FAIL",
                       ALL_OK : "JOB_QUEUE_ALL_OK",
                       ALL_FAIL : "JOB_QUEUE_ALL_FAIL",
                       USER_KILLED : "JOB_QUEUE_USER_KILLED"}
    
    

    

    def __init__(self, name, statistics=None):
        self.name = name
        self.status = Simulation.NOT_ACTIVE
        self.statuslog = []
        self.statistics = statistics

        self.resetTime()

    def checkStatus(self, type):
        return self.status == type

    def isWaiting(self):
        return self.checkStatus(Simulation.WAITING) or self.checkStatus(Simulation.PENDING)

    def isRunning(self):
        return self.checkStatus(Simulation.RUNNING)

    def hasFailed(self):
        return self.checkStatus(Simulation.ALL_FAIL)

    def notActive(self):
        return self.checkStatus(Simulation.NOT_ACTIVE)

    def finishedSuccessfully(self):
        return self.checkStatus(Simulation.ALL_OK)

    def isUserKilled(self):
        return self.checkStatus(Simulation.USER_KILLED)


    def setStatus(self, status):
        if len(self.statuslog) == 0 or not self.statuslog[len(self.statuslog) - 1] == status:
            self.statuslog.append(status)

            if status == Simulation.ALL_OK:
                self.setFinishedTime(int(time.time()))

        self.status = status

    def setStartTime(self, secs):
        self.startTime = secs

    def setSubmitTime(self, secs):
        self.submitTime = secs
        if self.submitTime > self.finishedTime:
            self.finishedTime = -1

    def setFinishedTime(self, secs):
        self.finishedTime = secs
        
        if not self.statistics is None:
            self.statistics.addTime(self.submitTime, self.startTime, self.finishedTime)

    def printTime(self, secs):
        if not secs == -1:
            print time.localtime(secs)

    def resetTime(self):
       self.startTime = -1
       self.submitTime = -1
       self.finishedTime = -1

    def __str__(self):
        return str(self.name)

    def __ge__(self, other):
        return self.name >= other.name

    def __lt__(self, other):
        return not self >= other


class SimulationStatistics:
    """A class that tracks statistics for Simulations (running time, waiting time, estimates, etc...)"""


    def __init__(self, name="default"):
        """Create a new tracking object"""
        self.name = name
        self.clear()
        self.old_job_count = 0
        self.old_duration = 0

    def clear(self):
        """Reset all values and stop estimate calculation"""
        self.jobs = 0
        self.waiting = 0
        self.running = 0
        self.total = 0
        self.start = 0
        self.last = 0

        self.stopped = True

    def startTiming(self):
        """Starts estimate calculation"""
        self.stopped = False
        self.start = int(time.time())

    def jobsPerSecond(self):
        """Returns the number of jobs per second as a float"""
        #t = int(time.time()) - self.start
        t = self.last - self.start
        if t > 0:
            return self.jobs / t
        else:
            return 0

    def averageRunningTime(self):
        """Calculates the average running time"""
        return self.running / self.jobs

    def secondsPerJob(self):
        """Returns how long a job takes in seconds"""
        return 1.0 / self.jobsPerSecond()

    def averageConcurrentJobs(self):
        """Returns the average number of jobs performed in parallel"""
        return max(self.running / (self.last - self.start), 1)

    def estimate(self, jobs):
        """Returns an estimate on how long the rest of the job will take. Jobs = the total number of jobs"""
        if self.jobsPerSecond() > 0:
            avg_concurrent_jobs = self.averageConcurrentJobs()
            avg_running = self.averageRunningTime()

            jobs_left = jobs - self.jobs
            est_remaining_running = avg_running * (jobs_left) / avg_concurrent_jobs
            timeUsed = int(time.time()) - self.last
            return est_remaining_running - timeUsed 
        else:
            return -1


    def addTime(self, submit, start, finish):
        """Add new statistical data to the tracker"""
        if not self.stopped:
            self.jobs += 1
            self.waiting += start - submit
            self.running += finish - start
            self.total += finish - submit
            self.last = int(time.time())

    def stop(self):
        """Pause the tracker. Estimate data will be reset"""
        self.old_job_count += self.jobs
        self.old_duration += int(time.time()) - self.start
        self.clear()

