from PyQt4 import QtGui, QtCore
from helpedwidget import *
from widgets.tablewidgets import AddRemoveWidget
import util
from widgets.pathchooser import PathChooser
from widgets.combochoice import ComboChoice
import widgets.stringbox

class ParameterPanel(HelpedWidget):
    """Shows a widget for parameters. The data structure expected and sent to the getter and setter is an array of Parameters."""

    passiveColor = QtGui.QColor(194, 194, 194)

    def __init__(self, parent=None, label="", help=""):
        """Construct a ParameterPanel."""
        HelpedWidget.__init__(self, parent, label, help)

        self.typeIcons = {}
        self.typeIcons["Field"] = util.resourceIcon("grid_16")
        self.typeIcons["Data"] = util.resourceIcon("data")
        self.typeIcons["Summary"] = util.resourceIcon("summary")
        self.typeIcons["Keyword"] = util.resourceIcon("key")

        self.addWidget((self.createListWidget()))


        self.pagesWidget = QtGui.QStackedWidget()


        self.emptyPanel = QtGui.QFrame(self)

        self.emptyPanel.setFrameShape(QtGui.QFrame.StyledPanel)
        self.emptyPanel.setFrameShadow(QtGui.QFrame.Plain)
        self.emptyPanel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.fieldPanel = FieldPanel(self)

        self.pagesWidget.addWidget(self.emptyPanel)
        self.pagesWidget.addWidget(self.fieldPanel)

        self.addWidget(self.pagesWidget)

        self.connect(self.list, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'), self.changeParameter)

        #self.addHelpButton()




    def changeParameter(self, current, previous):
        if not current:
            self.pagesWidget.setCurrentWidget(self.emptyPanel)
        elif current.getType() == "Field":
            self.pagesWidget.setCurrentWidget(self.fieldPanel)
            self.fieldPanel.setFieldModel(current.getData())
        else:
            self.pagesWidget.setCurrentWidget(self.emptyPanel)



    def presentSearch(self):
        """Is called to present the greyed out search"""
        self.disableSearch = True
        self.searchBox.setText("Search")
        palette = self.searchBox.palette()
        palette.setColor(self.searchBox.foregroundRole(), self.passiveColor)
        self.searchBox.setPalette(palette)

    def activateSearch(self):
        """Is called to remove the greyed out search"""
        self.disableSearch = False
        self.searchBox.setText("")
        palette = self.searchBox.palette()
        palette.setColor(self.searchBox.foregroundRole(), self.activeColor)
        self.searchBox.setPalette(palette)

    def enterSearch(self, focusEvent):
        """Called when the line edit gets the focus"""
        QtGui.QLineEdit.focusInEvent(self.searchBox, focusEvent)
        if str(self.searchBox.text()) == "Search":
            self.activateSearch()

    def exitSearch(self, focusEvent):
        """Called when the line edit looses focus"""
        QtGui.QLineEdit.focusOutEvent(self.searchBox, focusEvent)
        if str(self.searchBox.text()) == "":
            self.presentSearch()


    def searchInList(self, value):
        """Called when the contents of the search box changes"""
        if not self.disableSearch:
            for index in range(self.list.count()):
                param = self.list.item(index)
                if not param.getName().find(value) == -1:
                    param.setHidden(False)
                else:
                    param.setHidden(True)

    def addToList(self, type, name):
        """Adds a new parameter to the list"""
        param = Parameter(type, self.typeIcons[type], name)

        if type == "Field":
            param.setData(FieldModel(name))

        self.list.addItem(param)
        self.list.setCurrentItem(param)
        return param


    def addItem(self):
        """Called by the add button to insert a new parameter"""
        uniqueNames = []
        for index in range(self.list.count()):
            uniqueNames.append(str(self.list.item(index).text()))

        pd = ParameterDialog(self, self.typeIcons, uniqueNames)
        if pd.exec_():
            self.addToList(pd.getType(), pd.getName())


        #self.contentsChanged()


    def removeItem(self):
        """Called by the remove button to remove a selected parameter"""
        currentRow = self.list.currentRow()

        if currentRow >= 0:
            doDelete = QtGui.QMessageBox.question(self, "Delete parameter?", "Are you sure you want to delete the parameter?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

            if doDelete == QtGui.QMessageBox.Yes:
                self.list.takeItem(currentRow)
                #self.contentsChanged()


    def contentsChanged(self):
        """Called whenever the contents of a cell changes."""
        rowValues = []

        for rowIndex in range(self.table.rowCount()):
            row = []
            for columnIndex in range(self.table.columnCount()):
                item = self.table.item(rowIndex, columnIndex)
                if not item == None:
                    row.append(str(item.text()))
                else:
                    row.append("")

            rowValues.append(row)


        self.updateContent(rowValues)


    def fetchContent(self):
        """Retrieves data from the model and inserts it into the table"""
#        rows = self.getFromModel()
#
#        self.table.clear()
#        self.table.setHorizontalHeaderLabels(self.headers)
#
#        rowIndex = 0
#        for row in rows:
#            self.table.insertRow(rowIndex)
#            columnIndex = 0
#            for value in row:
#                item = QtGui.QTableWidgetItem(str(value))
#                self.table.setItem(rowIndex, columnIndex, item)
#                columnIndex+=1
#
#            rowIndex+=1


    def createListWidget(self):
        listWidget = QtGui.QWidget(self)
        listWidget.setMaximumWidth(130)
        listWidget.setMinimumWidth(130)
        vlayout = QtGui.QVBoxLayout()
        vlayout.setMargin(0)
        self.searchBox = QtGui.QLineEdit()
        self.searchBox.setToolTip("Type to search!")
        self.searchBox.focusInEvent = lambda event : self.enterSearch(event)
        self.searchBox.focusOutEvent = lambda event : self.exitSearch(event)
        self.activeColor = self.searchBox.palette().color(self.searchBox.foregroundRole())
        self.disableSearch = True
        self.presentSearch()
        self.connect(self.searchBox, QtCore.SIGNAL('textChanged(QString)'), self.searchInList)
        vlayout.addWidget(self.searchBox)
        self.list = QtGui.QListWidget(self)
        self.list.setMaximumWidth(128)
        self.list.setMinimumWidth(128)
        self.list.setMinimumHeight(350)
        self.list.setSortingEnabled(True)
        vlayout.addWidget(self.list)
        vlayout.addWidget(AddRemoveWidget(self, self.addItem, self.removeItem, True))
        listWidget.setLayout(vlayout)
        return listWidget



class FieldModel:
    type = "General"
    name = ""
    min = ""
    max = ""
    init = "None"
    output = "None"
    eclipse_file = ""
    init_files = ""
    file_generated_by_enkf = ""
    file_loaded_by_enkf = ""

    def __init__(self, name):
        self.name = name

                   
class FieldPanel(QtGui.QFrame):
    fieldModel = FieldModel("")

    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)

        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)


        self.fieldType = ComboChoice(self, ["Dynamic", "Parameter", "General"], "", "param_type")
        self.fieldType.setter = lambda model, value: self.typeChanged(value)
        self.fieldType.getter = lambda model: self.fieldModel.type

        self.min = widgets.stringbox.DoubleBox(self, "", "param_min")
        self.min.setter = lambda model, value: setattr(self.fieldModel, "min", value)
        self.min.getter = lambda model: self.fieldModel.min


        self.max = widgets.stringbox.DoubleBox(self, "", "param_max")
        self.max.setter = lambda model, value: setattr(self.fieldModel, "max", value)
        self.max.getter = lambda model: self.fieldModel.max


        self.init = ComboChoice(self, ["None", "EXP", "LOG", "POW10", "ADD", "MUL", "RANDINT", "RANDFLOAT"], "", "param_init")
        self.init.setter = lambda model, value: setattr(self.fieldModel, "init", value)
        self.init.getter = lambda model: self.fieldModel.init

        self.output = ComboChoice(self, ["None", "EXP", "LOG", "POW10", "ADD", "MUL", "RANDINT", "RANDFLOAT"], "", "param_output")
        self.output.setter = lambda model, value: setattr(self.fieldModel, "output", value)
        self.output.getter = lambda model: self.fieldModel.output

        self.eclipse_file = PathChooser(self, "", "param_eclipse_file", True)
        self.eclipse_file.setter = lambda model, value: setattr(self.fieldModel, "eclipse_file", value)
        self.eclipse_file.getter = lambda model: self.fieldModel.eclipse_file

        self.init_files = PathChooser(self, "", "param_init_files", True)
        self.init_files.setter = lambda model, value: setattr(self.fieldModel, "init_files", value)
        self.init_files.getter = lambda model: self.fieldModel.init_files

        self.file_generated_by_enkf = PathChooser(self, "", "param_file_generated_by_enkf", True)
        self.file_generated_by_enkf.setter = lambda model, value: setattr(self.fieldModel, "file_generated_by_enkf", value)
        self.file_generated_by_enkf.getter = lambda model: self.fieldModel.file_generated_by_enkf

        self.file_loaded_by_enkf = PathChooser(self, "", "param_file_loaded_by_enkf", True)
        self.file_loaded_by_enkf.setter = lambda model, value: setattr(self.fieldModel, "file_loaded_by_enkf", value)
        self.file_loaded_by_enkf.getter = lambda model: self.fieldModel.file_loaded_by_enkf

        layout.addRow("Field type:", self.fieldType)
        layout.addRow("Min:", self.min)
        layout.addRow("Max:", self.max)
        layout.addRow("Init:", self.init)
        layout.addRow("Output:", self.output)
        layout.addRow("Eclipse file:", self.eclipse_file)
        layout.addRow("Init files:", self.init_files)
        layout.addRow("File generated by EnKF:", self.file_generated_by_enkf)
        layout.addRow("File loaded by EnKF:", self.file_loaded_by_enkf)

        self.setLayout(layout)

        self.typeChanged("Dynamic")


    def typeChanged(self, value):
        setattr(self.fieldModel, "type", value)

        self.min.setEnabled(True)
        self.max.setEnabled(True)
        self.init.setEnabled(True)
        self.output.setEnabled(True)
        self.eclipse_file.setEnabled(True)
        self.init_files.setEnabled(True)
        self.file_generated_by_enkf.setEnabled(True)
        self.file_loaded_by_enkf.setEnabled(True)

        if value == "Dynamic":
            self.init.setEnabled(False)
            self.output.setEnabled(False)
            self.eclipse_file.setEnabled(False)
            self.init_files.setEnabled(False)
            self.file_generated_by_enkf.setEnabled(False)
            self.file_loaded_by_enkf.setEnabled(False)

        elif value == "Parameter":
            self.file_generated_by_enkf.setEnabled(False)
            self.file_loaded_by_enkf.setEnabled(False)

    def setFieldModel(self, fieldModel):
        self.fieldModel = fieldModel

        self.fieldType.fetchContent()
        self.min.fetchContent()
        self.max.fetchContent()
        self.init.fetchContent()
        self.output.fetchContent()
        self.eclipse_file.fetchContent()
        self.init_files.fetchContent()
        self.file_generated_by_enkf.fetchContent()
        self.file_loaded_by_enkf.fetchContent()



class ParameterDialog(QtGui.QDialog):

    invalidColor = QtGui.QColor(255, 235, 235)

    def __init__(self, parent, types, uniqueNames):
        QtGui.QDialog.__init__(self, parent)
        self.setModal(True)
        self.setWindowTitle('Create new parameter')

        self.uniqueNames = uniqueNames

        layout = QtGui.QFormLayout()

        layout.addRow(QtGui.QLabel("Select type and enter name of parameter:"))

        layout.addRow(self.createSpace())

        self.paramCombo = QtGui.QComboBox(self)

        keys = types.keys()
        keys.sort()
        for key in keys:
            self.paramCombo.addItem(types[key], key)
            
        layout.addRow("Type:", self.paramCombo)

        self.paramName = QtGui.QLineEdit(self)
        self.connect(self.paramName, QtCore.SIGNAL('textChanged(QString)'), self.validateName)
        self.validColor = self.paramName.palette().color(self.paramName.backgroundRole())

        layout.addRow("Name:", self.paramName)

        layout.addRow(self.createSpace())
        
        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        self.okbutton = buttons.button(QtGui.QDialogButtonBox.Ok)
        self.okbutton.setEnabled(False)
        
        layout.addRow(buttons)


        self.connect(buttons, QtCore.SIGNAL('accepted()'), self.accept)
        self.connect(buttons, QtCore.SIGNAL('rejected()'), self.reject)

        self.setLayout(layout)

    def notValid(self, msg):
        """Called when the parameter name is not valid."""
        self.okbutton.setEnabled(False)
        palette = self.paramName.palette()
        palette.setColor(self.paramName.backgroundRole(), self.invalidColor)
        self.paramName.setToolTip(msg)
        self.paramName.setPalette(palette)

    def valid(self):
        """Called when the parameter name is valid."""
        self.okbutton.setEnabled(True)
        palette = self.paramName.palette()
        palette.setColor(self.paramName.backgroundRole(), self.validColor)
        self.paramName.setToolTip("")
        self.paramName.setPalette(palette)


    def validateName(self, value):
        """Called to perform validation of a parameter name"""
        value = str(value)
        
        if value == "":
            self.notValid("Can not be empty!")
        elif not value.find(" ") == -1:
            self.notValid("No spaces allowed!")
        elif value in self.uniqueNames:
            self.notValid("Name must be unique!")
        else:
            self.valid()


    def createSpace(self):
        """Create some space in the layout"""
        space = QtGui.QFrame()
        space.setMinimumSize(QtCore.QSize(10, 10))
        return space

    def getType(self):
        """Return the type selected by the user"""
        return str(self.paramCombo.currentText())

    def getName(self):
        """Return the parameter name chosen by the user"""
        return str(self.paramName.text())


class Parameter(QtGui.QListWidgetItem):
    type = ""
    name = ""
    data = None

    def __init__(self, type, icon, name):
        QtGui.QListWidgetItem.__init__(self, icon, name)
        self.type = type
        self.name = name

    def getType(self):
        return self.type

    def getName(self):
        return self.name

    def __ge__(self, other):
        if self.getType() == other.getType():
            return self.getName().lower() >= other.getName().lower()
        else:
            return self.getType() >= other.getType()

    def __lt__(self, other):
        return not self >= other

    def setData(self, data):
        self.data = data

    def getData(self):
        return self.data