from PyQt4 import QtGui, QtCore
from widgets.tablewidgets import KeywordList
from widgets.validateddialog import ValidatedDialog
import ertwrapper
from widgets.combochoice import ComboChoice

# e = enkf_main_get_ensemble_config( enkf_main )
# s = ensemble_config_alloc_keylist_from_var_type(e , 1 # PARAMETER value from enkf_types.h)
# # Itererer over stringlist
# stringlist_free( s )
# range = enkf_main_get_ensemble_size( enkf_main )
#
#
# sl = stringlist_alloc_new()
# stringlist_append_copy(sl , "STRING")
#
# 
#
# enkf_main_initialize(enkf_main , sl , iens1 , iens2);
# stringlist_free( sl )
from widgets.helpedwidget import HelpedWidget
from widgets.util import resourceIcon

class ParametersAndMembers(HelpedWidget):


    def __init__(self, parent = None):
        HelpedWidget.__init__(self, parent)


        self.sourceCase = QtGui.QComboBox(self)
        self.sourceCase.setMaximumWidth(150)
        self.sourceCase.setToolTip("Select source case")
        self.sourceType = QtGui.QComboBox(self)
        self.sourceType.setMaximumWidth(100)
        self.sourceType.setToolTip("Select source type")
        self.sourceType.addItem("Analyzed")
        self.sourceType.addItem("Forecasted")
        self.sourceReportStep = QtGui.QComboBox(self)
        self.sourceReportStep.setMaximumWidth(125)
        self.sourceReportStep.setEditable(True)
        self.sourceReportStep.setValidator(QtGui.QIntValidator())
        self.sourceReportStep.addItem("Initial (0)")
        self.sourceReportStep.addItem("Prediction (n-1)")

        def checkSomething(event):
            QtGui.QComboBox.focusOutEvent(self.sourceReportStep, event)

            timestepMakesSense = False
            currentText = str(self.sourceReportStep.currentText())
            if currentText.startswith("Initial") or currentText.startswith("Prediction"):
                timestepMakesSense = True
            elif currentText.isdigit():
                intValue = int(currentText)
                if intValue >= 0 and intValue < 10:
                    timestepMakesSense = True

            if not timestepMakesSense:
                self.sourceReportStep.setCurrentIndex(0)


        self.sourceReportStep.focusOutEvent = lambda event : checkSomething(event)

        arrow = QtGui.QLabel(self)
        arrow.setPixmap(resourceIcon("arrow_right").pixmap(16, 16, QtGui.QIcon.Disabled))
        arrow.setMaximumSize(16, 16)

        self.targetType = QtGui.QComboBox(self)
        self.targetType.setMaximumWidth(100)
        self.targetType.setToolTip("Select target type")
        self.targetType.addItem("Analyzed")
        self.targetType.addItem("Forecasted")
        self.targetReportStep = QtGui.QSpinBox(self)
        self.targetReportStep.setMaximumWidth(55)
        self.targetReportStep.setMinimum(0)
        self.targetReportStep.setMaximum(0)

        stLayout = QtGui.QHBoxLayout()

        self.copyLabel = QtGui.QLabel("Copy:")
        self.copyLabel.setMaximumWidth(self.copyLabel.fontMetrics().width(self.copyLabel.text()))
        stLayout.addWidget(self.copyLabel)
        stLayout.addWidget(self.sourceCase)
        stLayout.addWidget(self.sourceType)
        stLayout.addWidget(self.sourceReportStep)
        stLayout.addWidget(arrow)
        stLayout.addWidget(self.targetType)
        stLayout.addWidget(self.targetReportStep)


        radioLayout = self.createRadioButtons()
        listLayout = self.createParameterMemberPanel()

        layout = QtGui.QVBoxLayout()
        layout.addLayout(radioLayout)
        layout.addLayout(listLayout)
        layout.addLayout(stLayout)

        self.addLayout(layout)

    def toggleCopyState(self, state):
        self.copyLabel.setEnabled(state)
        self.sourceCase.setEnabled(state)
        self.sourceType.setEnabled(state)
        self.sourceReportStep.setEnabled(state)
        self.targetType.setEnabled(state)
        self.targetReportStep.setEnabled(state)

    def fetchContent(self):
        data = self.getFromModel()

        self.parametersList.clear()
        self.membersList.clear()

        for parameter in data["parameters"]:
            self.parametersList.addItem(parameter)

        for member in data["members"]:
            self.membersList.addItem(str(member))


    def getter(self, ert):
        PARAMETER = 1 #PARAMETER value from enkf_types.h
        keylist = ert.enkf.ensemble_config_alloc_keylist_from_var_type(ert.ensemble_config, PARAMETER)
        parameters = ert.getStringList(keylist)
        ert.freeStringList(keylist)

        members = ert.enkf.enkf_main_get_ensemble_size(ert.main)

        return {"parameters" : parameters, "members" : range(members)}

    def initialize(self, ert):
        ert.setTypes("ensemble_config_alloc_keylist_from_var_type", ertwrapper.c_long, ertwrapper.c_int)
        ert.setTypes("enkf_main_get_ensemble_size", ertwrapper.c_int)

    def setter(self, ert, value):
        pass

    def createCheckPanel(self, checkall, uncheckall):
        self.checkAll = QtGui.QToolButton(self)
        self.checkAll.setIcon(resourceIcon("checked"))
        self.checkAll.setIconSize(QtCore.QSize(16, 16))
        self.checkAll.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.checkAll.setAutoRaise(True)
        self.checkAll.setToolTip("Select all")

        self.uncheckAll = QtGui.QToolButton(self)
        self.uncheckAll.setIcon(resourceIcon("notchecked"))
        self.uncheckAll.setIconSize(QtCore.QSize(16, 16))
        self.uncheckAll.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.uncheckAll.setAutoRaise(True)
        self.uncheckAll.setToolTip("Unselect all")

        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.setMargin(0)
        buttonLayout.setSpacing(0)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.checkAll)
        buttonLayout.addWidget(self.uncheckAll)

        self.connect(self.checkAll, QtCore.SIGNAL('clicked()'), checkall)
        self.connect(self.uncheckAll, QtCore.SIGNAL('clicked()'), uncheckall)

        return buttonLayout


    def createRadioButtons(self):

        radioLayout = QtGui.QVBoxLayout()
        self.toggleScratch = QtGui.QRadioButton("Initialize from scratch")
        radioLayout.addWidget(self.toggleScratch)
        self.toggleCopy = QtGui.QRadioButton("Copy")
        radioLayout.addWidget(self.toggleCopy)

        self.connect(self.toggleScratch, QtCore.SIGNAL('toggled(bool)'), lambda : self.toggleCopyState(self.toggleCopy.isChecked()))
        self.connect(self.toggleCopy, QtCore.SIGNAL('toggled(bool)'), lambda : self.toggleCopyState(self.toggleCopy.isChecked()))

        self.toggleScratch.toggle()
        return radioLayout

    def createParameterMemberPanel(self):
        self.parametersList = QtGui.QListWidget(self)
        self.parametersList.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.membersList = QtGui.QListWidget(self)
        self.membersList.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        parameterLayout = QtGui.QVBoxLayout()
        parameterLayout.addWidget(QtGui.QLabel("Parameters"))
        parameterLayout.addWidget(self.parametersList)
        parameterLayout.addLayout(self.createCheckPanel(self.parametersList.selectAll, self.parametersList.clearSelection))
        memberLayout = QtGui.QVBoxLayout()
        memberLayout.addWidget(QtGui.QLabel("Members"))
        memberLayout.addWidget(self.membersList)
        memberLayout.addLayout(self.createCheckPanel(self.membersList.selectAll, self.membersList.clearSelection))
        listLayout = QtGui.QHBoxLayout()
        listLayout.addLayout(parameterLayout)
        listLayout.addLayout(memberLayout)
        return listLayout

class InitPanel(QtGui.QFrame):
    
    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.Panel)
        self.setFrameShadow(QtGui.QFrame.Raised)

        initPanelLayout = QtGui.QVBoxLayout()
        self.setLayout(initPanelLayout)

        casePanel = QtGui.QFormLayout()


        def get_case_list(ert):
            fs = ert.enkf.enkf_main_get_fs(ert.main)
            caseList = ert.enkf.enkf_fs_alloc_dirlist(fs)

            list = ert.getStringList(caseList)
            ert.freeStringList(caseList)
            return list

        self.get_case_list = get_case_list # convenience: used by several functions


        casePanel.addRow("Current case:", self.createCurrentCaseCombo())
        casePanel.addRow("Cases:", self.createCaseList())



        parametersPanelLayout = QtGui.QHBoxLayout()


        parametersPanelLayout.addWidget(ParametersAndMembers(self))



        initPanelLayout.addLayout(casePanel)
        initPanelLayout.addWidget(self.createSeparator())
        initPanelLayout.addLayout(parametersPanelLayout)

        


    def createCaseList(self):
        """Creates a list that enables the creation of new cases. Removal has been disabled."""
        cases = KeywordList(self, "", "case_list")

        cases.newKeywordPopup = lambda list : ValidatedDialog(cases, "New case", "Enter name of new case:", list).showAndTell()
        cases.addRemoveWidget.enableRemoveButton(False)  #todo: add support for removal
        cases.list.setMaximumHeight(150)

        cases.initialize = lambda ert : [ert.setTypes("enkf_main_get_fs"),
                                         ert.setTypes("enkf_fs_alloc_dirlist"),
                                         ert.setTypes("enkf_fs_has_dir", ertwrapper.c_int),
                                         ert.setTypes("enkf_fs_select_write_dir", None)]


        def create_case(ert, cases):
            fs = ert.enkf.enkf_main_get_fs(ert.main)

            for case in cases:
                if not ert.enkf.enkf_fs_has_dir(fs, case):
                    ert.enkf.enkf_fs_select_write_dir(fs, case, True)
                    break

            self.currentCase.updateList(self.get_case_list(ert))
            self.currentCase.fetchContent()

        cases.getter = self.get_case_list
        cases.setter = create_case

        return cases

    def createCurrentCaseCombo(self):
        """Creates the combo that enables selection of the current case"""
        self.currentCase = ComboChoice(self, ["none"])

        def initialize_cases(ert):
            ert.setTypes("enkf_main_get_fs")
            ert.setTypes("enkf_fs_get_read_dir", ertwrapper.c_char_p)
            ert.setTypes("enkf_fs_select_read_dir", None, ertwrapper.c_char_p)

            self.currentCase.updateList(self.get_case_list(ert))
            

        self.currentCase.initialize = initialize_cases

        def get_current_case(ert):
            fs = ert.enkf.enkf_main_get_fs(ert.main)
            currentCase = ert.enkf.enkf_fs_get_read_dir(fs)
            #print "The selected case is: " + currentCase
            return currentCase

        self.currentCase.getter = get_current_case

        def select_case(ert, case):
            case = str(case)
            #print "Selecting case: " + case
            if not case == "":
                fs = ert.enkf.enkf_main_get_fs(ert.main)
                ert.enkf.enkf_fs_select_read_dir(fs, case)

        self.currentCase.setter = select_case

        return self.currentCase

    def createSeparator(self):
        """Adds a separator line to the panel."""
        qw = QtGui.QWidget()
        qwl = QtGui.QVBoxLayout()
        qw.setLayout(qwl)

        qf = QtGui.QFrame()
        qf.setFrameShape(QtGui.QFrame.HLine)
        qf.setFrameShadow(QtGui.QFrame.Sunken)

        qwl.addSpacing(10)
        qwl.addWidget(qf)
        qwl.addSpacing(10)

        return qw