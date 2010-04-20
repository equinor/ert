from PyQt4 import QtGui, QtCore

from widgets.helpedwidget import *
from widgets.tablewidgets import AddRemoveWidget
from widgets import util
from widgets.pathchooser import PathChooser
from widgets.combochoice import ComboChoice
import widgets.stringbox
from fieldpanel import *
from parameterdialog import ParameterDialog

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

        self.connect(self.list, QtCore.SIGNAL('itemSelectionChanged(QListWidgetItem *, QListWidgetItem *)'), self.changeParameter)

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
        # todo: emit when a new field is added also make initandcopy listen -> self.modelEmit("casesUpdated()")


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