from PyQt4 import QtGui, QtCore

from widgets.helpedwidget import *
from widgets.tablewidgets import AddRemoveWidget
from widgets import util
from widgets.pathchooser import PathChooser
from widgets.combochoice import ComboChoice
import widgets.stringbox
from fieldpanel import *
from parameterdialog import ParameterDialog
from widgets.searchablelist import SearchableList
from pages.config.parameters.datapanel import DataPanel
from pages.config.parameters.keywordpanel import KeywordPanel
import widgets.util
from pages.config.parameters.parametermodels import SummaryModel, FieldModel, DataModel, KeywordModel

class ParameterPanel(HelpedWidget):
    """Shows a widget for parameters. The data structure expected and sent to the getter and setter is an array of Parameters."""

    def __init__(self, parent=None, label="", help=""):
        """Construct a ParameterPanel."""
        HelpedWidget.__init__(self, parent, label, help)

        self.typeIcons = {"Field": util.resourceIcon("grid_16"),
                          "Data": util.resourceIcon("data"),
                          "Summary": util.resourceIcon("summary"),
                          "Keyword": util.resourceIcon("key")}

        self.searchableList = SearchableList(converter=lambda item : item.getName(), list_width=175)
        self.addWidget(self.searchableList)


        self.pagesWidget = QtGui.QStackedWidget()

        self.emptyPanel = widgets.util.createEmptyPanel()

        self.fieldPanel = FieldPanel(self)
        self.dataPanel = DataPanel(self)
        self.keywordPanel = KeywordPanel(self)

        self.pagesWidget.addWidget(self.emptyPanel)
        self.pagesWidget.addWidget(self.fieldPanel)
        self.pagesWidget.addWidget(self.dataPanel)
        self.pagesWidget.addWidget(self.keywordPanel)

        self.addWidget(self.pagesWidget)

        self.connect(self.searchableList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem, QListWidgetItem)'), self.changeParameter)
        self.connect(self.searchableList, QtCore.SIGNAL('addItem(list)'), self.addItem)
        self.connect(self.searchableList, QtCore.SIGNAL('removeItem(list)'), self.removeItem)

        #self.addHelpButton()


    def changeParameter(self, current, previous):
        if not current:
            self.pagesWidget.setCurrentWidget(self.emptyPanel)
        elif current.getType() == "Field":
            self.pagesWidget.setCurrentWidget(self.fieldPanel)
            self.fieldPanel.setFieldModel(current.getData())
        elif current.getType() == "Data":
            self.pagesWidget.setCurrentWidget(self.dataPanel)
            self.dataPanel.setDataModel(current.getData())
        elif current.getType() == "Keyword":
            self.pagesWidget.setCurrentWidget(self.keywordPanel)
            self.keywordPanel.setKeywordModel(current.getData())
        else:
            self.pagesWidget.setCurrentWidget(self.emptyPanel)


    def addToList(self, list, type, name):
        """Adds a new parameter to the list"""
        param = Parameter(type, self.typeIcons[type], name)

        if type == "Field":
            param.setData(FieldModel(name))
        elif type == "Data":
            param.setData(DataModel(name))
        elif type == "Keyword":
            param.setData(KeywordModel(name))
        elif type == "Summary":
            param.setData(SummaryModel(name))

        list.addItem(param)
        list.setCurrentItem(param)
        return param


    def addItem(self, list):
        """Called by the add button to insert a new parameter"""
        uniqueNames = []
        for index in range(list.count()):
            uniqueNames.append(str(list.item(index).text()))

        pd = ParameterDialog(self, self.typeIcons, uniqueNames)
        if pd.exec_():
            self.addToList(list, pd.getType(), pd.getName())


        #self.contentsChanged()
        # todo: emit when a new field is added also make initandcopy listen -> self.modelEmit("casesUpdated()")


    def removeItem(self, list):
        """Called by the remove button to remove a selected parameter"""
        currentRow = list.currentRow()

        if currentRow >= 0:
            doDelete = QtGui.QMessageBox.question(self, "Delete parameter?", "Are you sure you want to delete the parameter?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

            if doDelete == QtGui.QMessageBox.Yes:
                list.takeItem(currentRow)
                #self.contentsChanged()


    def contentsChanged(self):
        """Called whenever the contents of a cell changes."""
        pass


    def fetchContent(self):
        """Retrieves data from the model and inserts it into the list"""
        parameters = self.getFromModel()

        for parameter in parameters:
            ptype = "?"
            if parameter.TYPE == SummaryModel.TYPE:
                ptype = "Summary"
            elif parameter.TYPE == FieldModel.TYPE:
                ptype = "Field"
            elif parameter.TYPE == DataModel.TYPE:
                ptype = "Data"
            elif parameter.TYPE == KeywordModel.TYPE:
                ptype = "Keyword"

            param = Parameter(ptype, self.typeIcons[ptype], parameter.name)
            param.setData(parameter)

            self.searchableList.getList().addItem(param)
            self.searchableList.getList().setCurrentItem(param)

        if self.searchableList.getList().count > 0:
            self.searchableList.getList().setCurrentRow(0)


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