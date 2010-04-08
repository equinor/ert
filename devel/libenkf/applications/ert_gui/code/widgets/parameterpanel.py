from PyQt4 import QtGui, QtCore
from helpedwidget import *
from widgets.tablewidgets import AddRemoveWidget
import util

class ParameterPanel(HelpedWidget):
    """Shows a widget for parameters. The data structure expected and sent to the getter and setter is an array of Parameters."""

    passiveColor = QtGui.QColor(194, 194, 194)

    def __init__(self, parent=None, label="", help=""):
        """Construct a ParameterPanel."""
        HelpedWidget.__init__(self, parent, label, help)

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
        self.addWidget(listWidget)


        self.pagesWidget = QtGui.QStackedWidget()

        panel = QtGui.QFrame()
        panel.setFrameShape(QtGui.QFrame.Panel)
        panel.setFrameShadow(QtGui.QFrame.Raised)
        panel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.pagesWidget.addWidget(panel)

        self.addWidget(self.pagesWidget)


        self.addHelpButton()

        self.typeIcons = {}
        self.typeIcons["Field"] = util.resourceIcon("grid_16")
        self.typeIcons["Data"] = util.resourceIcon("data")
        self.typeIcons["Summary"] = util.resourceIcon("summary")
        self.typeIcons["Keyword"] = util.resourceIcon("key")



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
        self.list.addItem(param)
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