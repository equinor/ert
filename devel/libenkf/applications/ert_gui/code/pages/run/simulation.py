from PyQt4 import QtGui, QtCore
from widgets.util import resourceIcon
import time

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

    size = QtCore.QSize(32, 18)

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

        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter.drawRect(rect)

        if option.state & QtGui.QStyle.State_Selected:
            painter.fillRect(option.rect, QtGui.QColor(255, 255, 255, 164))

        painter.drawText(rect, QtCore.Qt.AlignCenter + QtCore.Qt.AlignVCenter, str(data.name))

        painter.restore()

    def sizeHint(self, option, index):
        return self.size


class SimulationPanel(QtGui.QFrame):
    def __init__(self, parent=None):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.Panel)
        self.setFrameShadow(QtGui.QFrame.Raised)

        self.setMinimumWidth(150)
        self.setMaximumWidth(150)

        layout = QtGui.QVBoxLayout()

        self.killButton = QtGui.QToolButton(self)
        self.killButton.setIcon(resourceIcon("cross"))
        self.killButton.setToolTip("Kill job")
        self.connect(self.killButton, QtCore.SIGNAL('clicked()'), self.action)
        
        self.restartButton = QtGui.QToolButton(self)
        self.restartButton.setIcon(resourceIcon("refresh"))
        self.restartButton.setToolTip("Restart job")
        self.connect(self.restartButton, QtCore.SIGNAL('clicked()'), self.action)
        
        self.rrButton = QtGui.QToolButton(self)
        self.rrButton.setIcon(resourceIcon("refresh_resample"))
        self.rrButton.setToolTip("Resample and restart")
        self.connect(self.rrButton, QtCore.SIGNAL('clicked()'), self.action)

        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addWidget(self.killButton)
        buttonLayout.addWidget(self.restartButton)
        buttonLayout.addWidget(self.rrButton)

        layout.addStretch(1)
        layout.addLayout(buttonLayout)

        self.setLayout(layout)

    def action(self):
        print "Woohoo"

    def setSimulation(self, selection=None):
        pass

    def setModel(self, ert):
        self.ert = ert

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
        
        self.startTime = -1
        self.submitTime = -1
        self.finishedTime = -1

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

    def isUserKilled(self):
        return self.checkStatus("JOB_QUEUE_USER_KILLED")


    def setStatus(self, status):
        if len(self.statuslog) == 0 or not self.statuslog[len(self.statuslog) - 1] == status:
            self.statuslog.append(status)

            if status == self.job_status_type_reverse["JOB_QUEUE_ALL_OK"]:
                self.finishedTime = int(time.time())
                print self.startTime, self.submitTime, self.finishedTime

        self.status = status

    def setStartTime(self, secs):
        self.startTime = secs
        #self.printTime(secs)

    def setSubmitTime(self, secs):
        self.submitTime = secs
        #self.printTime(secs)

    def printTime(self, secs):
        if not secs == -1:
            print time.localtime(secs)
