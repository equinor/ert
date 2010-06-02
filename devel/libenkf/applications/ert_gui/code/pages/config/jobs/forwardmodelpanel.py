from widgets.helpedwidget import HelpedWidget
from widgets.searchablelist import SearchableList
from PyQt4 import QtGui, QtCore
from widgets.pathchooser import PathChooser
from widgets.validateddialog import ValidatedDialog
import widgets.util
import os
from widgets.util import ValidationInfo
from pages.config.jobs.jobsdialog import EditJobDialog
from widgets.stringbox import StringBox

class ForwardModelPanel(HelpedWidget):
    """
    Widget for adding, removing and editing forward models.
    These additional ContentModel functions must be implemented: insert and remove.
    The panel expects remove to return True or False based on the success of the removal.
    """

    def __init__(self, parent=None):
        HelpedWidget.__init__(self, parent, "Forward Model", "forward_model")

        self.forward_model = ForwardModel("undefined")

        self.createWidgets(parent)

        self.emptyPanel = widgets.util.createEmptyPanel()

        self.pagesWidget = QtGui.QStackedWidget()
        self.pagesWidget.addWidget(self.emptyPanel)
        self.pagesWidget.addWidget(self.forward_model_panel)
        self.addWidget(self.pagesWidget)

        self.addHelpButton()

    def createWidgets(self, parent):
        self.searchableList = SearchableList(parent, list_height=150, list_width=150, ignore_case=True, order_editable=True)
        self.addWidget(self.searchableList)
        self.connect(self.searchableList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem, QListWidgetItem)'),
                     self.changeParameter)
        self.connect(self.searchableList, QtCore.SIGNAL('addItem(list)'), self.addItem)
        self.connect(self.searchableList, QtCore.SIGNAL('removeItem(list)'), self.removeItem)
        self.connect(self.searchableList, QtCore.SIGNAL('orderChanged(list)'), self.forwardModelChanged)


        self.forward_model_panel = widgets.util.createEmptyPanel()

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self.forward_model_args = StringBox(self, "", "forward_model_arguments")
        self.forward_model_args.setter = self.setArguments
        self.forward_model_args.getter = lambda model: self.forward_model.arguments

        layout.addRow("Arguments:", self.forward_model_args)

        layout.addRow(widgets.util.createSpace(20))

        self.help_text = QtGui.QLabel()
        self.help_text.setText("")

        layout.addRow(widgets.util.centeredWidget(self.help_text))

        self.forward_model_panel.setLayout(layout)

    def setArguments(self, model, arguments):
        self.forward_model.setArguments(arguments)
        self.forwardModelChanged()

    def fetchContent(self):
        """Retrieves data from the model and inserts it into the widget"""
        forward_model = self.getFromModel()

#        for job in jobs:
#            jobitem = QtGui.QListWidgetItem()
#            jobitem.setText(job.name)
#            jobitem.setData(QtCore.Qt.UserRole, job)
#            jobitem.setToolTip(job.name)
#            self.searchableList.list.addItem(jobitem)

    def setForwardModel(self, forward_model):
        self.forward_model = forward_model
        self.help_text.setText(forward_model.help_text)
        self.forward_model_args.fetchContent()

    def changeParameter(self, current, previous):
        """Switch between forward models. Selection from the list"""
        if current is None:
            self.pagesWidget.setCurrentWidget(self.emptyPanel)
        else:
            self.pagesWidget.setCurrentWidget(self.forward_model_panel)
            self.setForwardModel(current.data(QtCore.Qt.UserRole).toPyObject())

    def forwardModelChanged(self):
        self.updateContent(self.searchableList.getItems())

    def addToList(self, list, name):
        """Adds a new job to the list"""
        param = QtGui.QListWidgetItem()
        param.setText(name)

        new_job = ForwardModel(name)
        param.setData(QtCore.Qt.UserRole, new_job)

        list.addItem(param)
        list.setCurrentItem(param)
        return new_job

    def addItem(self, list):
        """Called by the add button to insert a new job"""
        availableNames = []
        #todo: fetch list from ert

        pd = ValidatedDialog(self, "New forward model", "Select a job:", availableNames, True)
        if pd.exec_():
            self.addToList(list, pd.getName())
            self.forwardModelChanged()

    def removeItem(self, list):
        """Called by the remove button to remove a selected job"""
        currentRow = list.currentRow()

        if currentRow >= 0:
            title = "Delete forward model?"
            msg = "Are you sure you want to delete the forward model?"
            btns = QtGui.QMessageBox.Yes | QtGui.QMessageBox.No
            doDelete = QtGui.QMessageBox.question(self, title, msg, btns)

            if doDelete == QtGui.QMessageBox.Yes:
                list.takeItem(currentRow)
                self.forwardModelChanged()



class ForwardModel:

    def __init__(self, name, arguments=None, help_text="No help available for this job."):
        self.name = name
        if arguments is None:
            arguments = ""
        self.arguments = arguments
        self.help_text = help_text

    def setArguments(self, args):
        if args is None:
            args = ""
        self.arguments = args


