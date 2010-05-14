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
        elif FieldModel.TYPE == current.getType():
            self.pagesWidget.setCurrentWidget(self.fieldPanel)
            self.fieldPanel.setFieldModel(current.getData())
        elif DataModel.TYPE == current.getType():
            self.pagesWidget.setCurrentWidget(self.dataPanel)
            self.dataPanel.setDataModel(current.getData())
        elif KeywordModel.TYPE == current.getType():
            self.pagesWidget.setCurrentWidget(self.keywordPanel)
            self.keywordPanel.setKeywordModel(current.getData())
        else:
            self.pagesWidget.setCurrentWidget(self.emptyPanel)

    def addToList(self, list, type_name, name):
        """Adds a new parameter to the list"""

        if type_name == FieldModel.TYPE.name:
            type = FieldModel.TYPE
            data = FieldModel(name)
        elif type_name == DataModel.TYPE.name:
            type = DataModel.TYPE
            data = DataModel(name)
        elif type_name == KeywordModel.TYPE.name:
            type = KeywordModel.TYPE
            data = KeywordModel(name)
        elif type_name == SummaryModel.TYPE.name:
            type = SummaryModel.TYPE
            data = SummaryModel(name)
        else:
            raise AssertionError("Type name unknown: %s" % (type_name))

        param = Parameter(name, type)
        param.setData(data)

        list.addItem(param)
        list.setCurrentItem(param)
        return param


    def addItem(self, list):
        """Called by the add button to insert a new parameter"""
        uniqueNames = []
        for index in range(list.count()):
            uniqueNames.append(str(list.item(index).text()))

        pd = ParameterDialog(self, Parameter.typeIcons, uniqueNames)
        if pd.exec_():
            result = self.addToList(list, pd.getTypeName(), pd.getName())
            self.updateContent(result, operation=ContentModel.INSERT)

        # todo: emit when a new field is added also make initandcopy listen -> self.modelEmit("casesUpdated()")


    def removeItem(self, list):
        """Called by the remove button to remove a selected parameter"""
        currentRow = list.currentRow()

        if currentRow >= 0:
            doDelete = QtGui.QMessageBox.question(self, "Delete parameter?", "Are you sure you want to delete the parameter?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

            if doDelete == QtGui.QMessageBox.Yes:
                item = list.curentItem()
                self.updateContent(item.getName(), operation=ContentModel.REMOVE)
                list.takeItem(currentRow)


    def fetchContent(self):
        """Retrieves data from the model and inserts it into the list"""
        parameters = self.getFromModel()

        for parameter in parameters:
            if parameter is None:
                raise AssertionError("Unknown type name!")

            param = Parameter(parameter.name, parameter.TYPE)
            param.setData(parameter)

            self.searchableList.getList().addItem(param)
            self.searchableList.getList().setCurrentItem(param)

        if self.searchableList.getList().count > 0:
            self.searchableList.getList().setCurrentRow(0)


class Parameter(QtGui.QListWidgetItem):
    """ListWidgetItem class that represents a Parameter with an associated icon."""
    typeIcons = {FieldModel.TYPE: util.resourceIcon("grid_16"),
                 DataModel.TYPE: util.resourceIcon("data"),
                 SummaryModel.TYPE: util.resourceIcon("summary"),
                 KeywordModel.TYPE: util.resourceIcon("key")}

    def __init__(self, name, type):
        QtGui.QListWidgetItem.__init__(self, Parameter.typeIcons[type], name)
        self.type = type
        self.name = name
        self.data = None

    def getType(self):
        """Retruns the type of this parameter"""
        return self.type

    def getName(self):
        """Returns the name of this parameter (keyword)"""
        return self.name

    def __ge__(self, other):
        if self.type.name == other.type.name:
            return self.name.lower() >= other.name.lower()
        else:
            return self.type.name >= other.type.name

    def __lt__(self, other):
        return not self >= other

    def setData(self, data):
        """Set user data for this parameter."""
        self.data = data

    def getData(self):
        """Retrieve the user data."""
        return self.data