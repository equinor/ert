from PyQt4 import QtGui, QtCore
import ertwrapper

from widgets.helpedwidget import HelpedWidget, ContentModel
from widgets.util import resourceIcon, ListCheckPanel, ValidatedTimestepCombo

class RunWidget(HelpedWidget):

    def __init__(self, parent=None):
        HelpedWidget.__init__(self, parent, widgetLabel="", helpLabel="widget_run")

        self.membersList = QtGui.QListWidget(self)
        self.membersList.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)

        self.membersList.setViewMode(QtGui.QListView.IconMode)
        self.membersList.setMovement(QtGui.QListView.Static)
        self.membersList.setResizeMode(QtGui.QListView.Adjust)
        self.membersList.setGridSize(QtCore.QSize(32, 16))
        self.membersList.setSelectionRectVisible(False)


        #memberLayout = QtGui.QVBoxLayout()
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
        self.simulateTo = ValidatedTimestepCombo(parent, fromLabel="End of History", toLabel="End")
        memberLayout.addRow("Run simulation from: ", self.simulateFrom)
        memberLayout.addRow("Run simulation to: ", self.simulateTo)
        memberLayout.addRow("Mode: ", self.createRadioButtons())
        memberLayout.addRow(membersCheckPanel)
        memberLayout.addRow("Members:", self.membersList)




        self.addLayout(memberLayout)

        self.setRunpath("...")

        self.modelConnect("ensembleResized()", self.fetchContent)
        self.rbAssimilation.toggle()

    def setRunpath(self, runpath):
        #self.runpathLabel.setText("Runpath: " + runpath)
        self.runpathLabel.setText(runpath)

    def fetchContent(self):
        data = self.getFromModel()

        historyLength = data["history_length"]

        self.membersList.clear()

        for member in data["members"]:
            self.membersList.addItem("%03d" % (member))
            #self.membersList.addItem(str(member))

        self.simulateFrom.setHistoryLength(historyLength)
        self.simulateTo.setFromValue(historyLength)
        self.simulateTo.setToValue(-1)
        self.simulateTo.setMinTimeStep(0)
        self.simulateTo.setMaxTimeStep(historyLength)

        self.membersList.selectAll()
        

    def initialize(self, ert):
        ert.setTypes("enkf_main_get_ensemble_size", ertwrapper.c_int)
        ert.setTypes("enkf_main_get_fs")
        ert.setTypes("enkf_fs_get_read_dir", ertwrapper.c_char_p)
        ert.setTypes("enkf_fs_alloc_dirlist")
        ert.setTypes("enkf_main_get_history_length", ertwrapper.c_int)


    def getter(self, ert):
        members = ert.enkf.enkf_main_get_ensemble_size(ert.main)
        historyLength = ert.enkf.enkf_main_get_history_length(ert.main)

        return {"members" : range(members), "history_length" : historyLength}


    def rbToggle(self):
        print "Radio button toggled"

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

class RunPanel(QtGui.QFrame):
    
    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.Panel)
        self.setFrameShadow(QtGui.QFrame.Raised)


        panelLayout = QtGui.QVBoxLayout()
        self.setLayout(panelLayout)

                
        button = QtGui.QPushButton("Refetch")
        self.connect(button, QtCore.SIGNAL('clicked()'), ContentModel.updateObservers)

        panelLayout.addWidget(button)
        panelLayout.addWidget(RunWidget())




    def createSeparator(self):
        """Adds a separator line to the panel."""
        qw = QtGui.QWidget()
        qwl = QtGui.QVBoxLayout()
        qw.setLayout(qwl)

        qf = QtGui.QFrame()
        qf.setFrameShape(QtGui.QFrame.HLine)
        qf.setFrameShadow(QtGui.QFrame.Sunken)

        qwl.addSpacing(5)
        qwl.addWidget(qf)
        qwl.addSpacing(5)

        return qw
