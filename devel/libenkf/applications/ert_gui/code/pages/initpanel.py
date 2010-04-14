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

    listOfParameters = []
    listOfDynamicParameters = []

    def __init__(self, parent = None):
        HelpedWidget.__init__(self, parent)

        self.sourceCase = QtGui.QComboBox(self)
        self.sourceCase.setMaximumWidth(150)
        self.sourceCase.setMinimumWidth(150)
        self.sourceCase.setToolTip("Select source case")
        self.sourceType = QtGui.QComboBox(self)
        self.sourceType.setMaximumWidth(100)
        self.sourceType.setToolTip("Select source type")
        self.sourceType.addItem("Analyzed")
        self.sourceType.addItem("Forecasted")
        self.sourceReportStep = self.createValidatedTimestepCombo()
        self.sourceCompleteEnsembleCheck = QtGui.QCheckBox("Complete Ensemble")
        self.sourceCompleteEnsembleCheck.setChecked(True)
        self.connect(self.sourceCompleteEnsembleCheck, QtCore.SIGNAL('stateChanged(int)'), lambda state : self.toggleCompleteEnsembleState(self.sourceCompleteEnsembleCheck.isChecked()))


        self.maxTimeStep = 11

        self.targetType = QtGui.QComboBox(self)
        self.targetType.setMaximumWidth(100)
        self.targetType.setToolTip("Select target type")
        self.targetType.addItem("Analyzed")
        self.targetType.addItem("Forecasted")
        self.targetReportStep = self.createValidatedTimestepCombo()

        stLayout = QtGui.QGridLayout()
        stLayout.setColumnStretch(8, 1)

        stLayout.addWidget(QtGui.QLabel("Case"), 0, 1)
        stLayout.addWidget(QtGui.QLabel("State"), 0, 3)
        stLayout.addWidget(QtGui.QLabel("Timestep"), 0, 5)


        self.sourceLabel = QtGui.QLabel("Source:")
        stLayout.addWidget(self.sourceLabel, 1, 0)
        stLayout.addWidget(self.sourceCase, 1, 1)
        stLayout.addWidget(self.sourceType, 1, 3)
        stLayout.addWidget(self.sourceReportStep, 1, 5)
        stLayout.addWidget(self.sourceCompleteEnsembleCheck, 1, 7)

        self.targetCaseLabel = QtGui.QLabel("none?")
        font = self.targetCaseLabel.font()
        font.setWeight(QtGui.QFont.Bold)
        self.targetCaseLabel.setFont(font)
        self.targetCaseLabel.setMinimumWidth(self.sourceCase.minimumWidth())

        self.targetLabel = QtGui.QLabel("Target:")
        stLayout.addWidget(self.targetLabel, 2, 0)
        stLayout.addWidget(self.targetCaseLabel, 2, 1)
        stLayout.addWidget(self.targetType, 2, 3)
        stLayout.addWidget(self.targetReportStep, 2, 5)


        radioLayout = self.createRadioButtons()
        listLayout = self.createParameterMemberPanel()

        self.actionButton = QtGui.QPushButton("Initialize")
        actionLayout = QtGui.QHBoxLayout()
        actionLayout.addStretch(1)
        actionLayout.addWidget(self.actionButton)
        actionLayout.addStretch(1)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(radioLayout)
        layout.addSpacing(5)
        layout.addLayout(listLayout)
        layout.addSpacing(5)
        layout.addLayout(stLayout)
        layout.addSpacing(5)
        layout.addLayout(actionLayout)

        self.addLayout(layout)

        self.toggleScratch.toggle()

    def toggleCompleteEnsembleState(self, checkState):
        self.parametersList.setEnabled(not checkState)
        self.parametersList.checkAll.setEnabled(not checkState)
        self.parametersList.uncheckAll.setEnabled(not checkState)

        if checkState:
            self.parametersList.selectAll()


    def toggleActionState(self, action="Initialize", showCopyParameters = False, selectSource = False, selectTarget = False):
        self.sourceLabel.setEnabled(selectSource)
        self.sourceCase.setEnabled(selectSource)
        self.sourceType.setEnabled(selectSource)
        self.sourceReportStep.setEnabled(selectSource)
        self.sourceCompleteEnsembleCheck.setEnabled(showCopyParameters)

        if not selectSource:
            self.sourceReportStep.setCurrentIndex(0)

        self.targetLabel.setEnabled(selectTarget)
        self.targetCaseLabel.setEnabled(selectTarget)
        self.targetType.setEnabled(selectTarget)
        self.targetReportStep.setEnabled(selectTarget)


        if not selectTarget:
            self.targetReportStep.setCurrentIndex(0)

        self.actionButton.setText(action)


        self.parametersList.clear()
        self.parametersList.addItems(self.listOfParameters)

        self.parametersList.setEnabled(True)
        self.parametersList.checkAll.setEnabled(True)
        self.parametersList.uncheckAll.setEnabled(True)


        if showCopyParameters:
            self.parametersList.addItems(self.listOfDynamicParameters)
            self.toggleCompleteEnsembleState(self.sourceCompleteEnsembleCheck.isChecked())


        
    def fetchContent(self): #todo: add support for updated parameters and cases. Make other operations emit signals
        data = self.getFromModel()

        self.parametersList.clear()
        self.membersList.clear()
        self.sourceCase.clear()

        self.listOfParameters = data["parameters"]
        self.listOfDynamicParameters = data["dynamic_parameters"]

        for member in data["members"]:
            self.membersList.addItem(str(member))

        for case in data["cases"]:
            if not case == data["current_case"]:
                self.sourceCase.addItem(case)

        self.maxTimeStep = data["history_length"]

        self.sourceReportStep.setHistoryLength(self.maxTimeStep)
        self.targetReportStep.setHistoryLength(self.maxTimeStep)

        self.targetCaseLabel.setText(data["current_case"])

        if self.toggleScratch.isChecked():
            self.toggleScratch.emit(QtCore.SIGNAL("toggled(bool)"), True)
        elif self.toggleInitCopy.isChecked():
            self.toggleInitCopy.emit(QtCore.SIGNAL("toggled(bool)"), True)
        else:
            self.toggleCopy.emit(QtCore.SIGNAL("toggled(bool)"), True)

            
    def initialize(self, ert):
        ert.setTypes("ensemble_config_alloc_keylist_from_var_type", ertwrapper.c_long, ertwrapper.c_int)
        ert.setTypes("enkf_main_get_ensemble_size", ertwrapper.c_int)
        ert.setTypes("enkf_main_get_fs")
        ert.setTypes("enkf_fs_get_read_dir", ertwrapper.c_char_p)
        ert.setTypes("enkf_fs_alloc_dirlist")
        ert.setTypes("enkf_main_get_history_length", ertwrapper.c_int)


    def getter(self, ert):
        #enums from enkf_types.h
        PARAMETER     = 1
        DYNAMIC_STATE = 2

        keylist = ert.enkf.ensemble_config_alloc_keylist_from_var_type(ert.ensemble_config, PARAMETER )
        parameters = ert.getStringList(keylist)
        ert.freeStringList(keylist)

        keylist = ert.enkf.ensemble_config_alloc_keylist_from_var_type(ert.ensemble_config,  DYNAMIC_STATE )
        dynamicParameters = ert.getStringList(keylist)
        ert.freeStringList(keylist)

        members = ert.enkf.enkf_main_get_ensemble_size(ert.main)

        fs = ert.enkf.enkf_main_get_fs(ert.main)
        currentCase = ert.enkf.enkf_fs_get_read_dir(fs)

        caseList = ert.enkf.enkf_fs_alloc_dirlist(fs)
        list = ert.getStringList(caseList)
        ert.freeStringList(caseList)

        historyLength = ert.enkf.enkf_main_get_history_length(ert.main)

        return {"parameters" : parameters,
                "dynamic_parameters" : dynamicParameters,
                "members" : range(members),
                "current_case" : currentCase,
                "cases" : list,
                "history_length" : historyLength}


    def setter(self, ert, value):
        """The setting of these values are activated by a separate button."""
        pass


    def createCheckPanel(self, list):
        list.checkAll = QtGui.QToolButton(self)
        list.checkAll.setIcon(resourceIcon("checked"))
        list.checkAll.setIconSize(QtCore.QSize(16, 16))
        list.checkAll.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        list.checkAll.setAutoRaise(True)
        list.checkAll.setToolTip("Select all")

        list.uncheckAll = QtGui.QToolButton(self)
        list.uncheckAll.setIcon(resourceIcon("notchecked"))
        list.uncheckAll.setIconSize(QtCore.QSize(16, 16))
        list.uncheckAll.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        list.uncheckAll.setAutoRaise(True)
        list.uncheckAll.setToolTip("Unselect all")

        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.setMargin(0)
        buttonLayout.setSpacing(0)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(list.checkAll)
        buttonLayout.addWidget(list.uncheckAll)

        self.connect(list.checkAll, QtCore.SIGNAL('clicked()'), list.selectAll)
        self.connect(list.uncheckAll, QtCore.SIGNAL('clicked()'), list.clearSelection)

        return buttonLayout


    def createRadioButtons(self):
        radioLayout = QtGui.QVBoxLayout()
        radioLayout.setSpacing(2)
        self.toggleScratch = QtGui.QRadioButton("Initialize from scratch")
        radioLayout.addWidget(self.toggleScratch)
        self.toggleInitCopy = QtGui.QRadioButton("Initialize from existing case")
        radioLayout.addWidget(self.toggleInitCopy)
        self.toggleCopy = QtGui.QRadioButton("Copy from existing case")
        radioLayout.addWidget(self.toggleCopy)

        self.connect(self.toggleScratch, QtCore.SIGNAL('toggled(bool)'), lambda : self.toggleActionState())
        self.connect(self.toggleInitCopy, QtCore.SIGNAL('toggled(bool)'), lambda : self.toggleActionState(selectSource = True))
        self.connect(self.toggleCopy, QtCore.SIGNAL('toggled(bool)'), lambda : self.toggleActionState(action = "Copy", selectSource=True, showCopyParameters=True, selectTarget=True))

        return radioLayout


    def createParameterMemberPanel(self):
        self.parametersList = QtGui.QListWidget(self)
        self.parametersList.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.membersList = QtGui.QListWidget(self)
        self.membersList.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)

        parameterLayout = QtGui.QVBoxLayout()
        parametersCheckPanel = self.createCheckPanel(self.parametersList)
        parametersCheckPanel.insertWidget(0, QtGui.QLabel("Parameters"))
        parameterLayout.addLayout(parametersCheckPanel)
        parameterLayout.addWidget(self.parametersList)

        memberLayout = QtGui.QVBoxLayout()
        membersCheckPanel = self.createCheckPanel(self.membersList)
        membersCheckPanel.insertWidget(0, QtGui.QLabel("Members"))
        memberLayout.addLayout(membersCheckPanel)
        memberLayout.addWidget(self.membersList)

        listLayout = QtGui.QHBoxLayout()
        listLayout.addLayout(parameterLayout)
        listLayout.addLayout(memberLayout)

        return listLayout


    def createValidatedTimestepCombo(self):
        validatedCombo = QtGui.QComboBox(self)
        validatedCombo.setMaximumWidth(125)
        validatedCombo.setEditable(True)
        validatedCombo.setValidator(QtGui.QIntValidator())
        validatedCombo.addItem("Initial (0)")
        validatedCombo.addItem("Prediction (n-1)")


        def focusOutValidation(combo, maxTimeStep, event):
            QtGui.QComboBox.focusOutEvent(combo, event)

            timestepMakesSense = False
            currentText = str(combo.currentText())
            if currentText.startswith("Initial") or currentText.startswith("Prediction"):
                timestepMakesSense = True
            elif currentText.isdigit():
                intValue = int(currentText)
                timestepMakesSense = True

                if intValue > maxTimeStep:
                     combo.setCurrentIndex(1)


            if not timestepMakesSense:
                combo.setCurrentIndex(0)

        validatedCombo.focusOutEvent = lambda event : focusOutValidation(validatedCombo, self.maxTimeStep, event)


        def setHistoryLength(length):
            validatedCombo.setItemText(1, "Prediction (" + str(length) + ")")

        validatedCombo.setHistoryLength = setHistoryLength

        return validatedCombo



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
        self.currentCase.combo.setMinimumWidth(150)

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
