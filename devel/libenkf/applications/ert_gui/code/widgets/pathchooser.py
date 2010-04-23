import os
from PyQt4 import QtGui, QtCore
from helpedwidget import *
import re

class PathChooser(HelpedWidget):
    """PathChooser shows, enables choosing of and validates paths. The data structure expected and sent to the getter and setter is a string."""

    errorColor = QtGui.QColor(255, 235, 235)
    invalidColor = QtGui.QColor(235, 235, 255)

    file_does_not_exist_msg = "The specified path does not exist."
    required_field_msg = "A value is required."
    path_format_msg = "Must be a path format."

    def __init__(self, parent=None, pathLabel="Path", help="", files=False, must_be_set=True, path_format=False, must_exist=True):
        """Construct a PathChooser widget"""
        HelpedWidget.__init__(self, parent, pathLabel, help)

        self.editing = True
        self.selectFiles = files
        self.must_be_set = must_be_set
        self.path_format = path_format
        self.must_exist = must_exist

        self.pathLine = QtGui.QLineEdit()
        #self.pathLine.setMinimumWidth(250)
        
        self.connect(self.pathLine, QtCore.SIGNAL('editingFinished()'), self.validatePath)
        self.connect(self.pathLine, QtCore.SIGNAL('editingFinished()'), self.contentsChanged)
        self.connect(self.pathLine, QtCore.SIGNAL('textChanged(QString)'), self.validatePath)
        self.addWidget(self.pathLine)

        dialogButton = QtGui.QToolButton(self)
        dialogButton.setIcon(resourceIcon("folder"))
        dialogButton.setIconSize(QtCore.QSize(16, 16))
        self.connect(dialogButton, QtCore.SIGNAL('clicked()'), self.selectDirectory)
        self.addWidget(dialogButton)

        #self.addStretch()
        self.addHelpButton()

        self.validColor = self.pathLine.palette().color(self.pathLine.backgroundRole())

        #self.pathLine.setText(os.path.expanduser("~"))
        self.pathLine.setText(os.getcwd())

        self.editing = False



    def validatePath(self):
        """Called whenever the path is modified"""
        palette = self.pathLine.palette()

        text = str(self.pathLine.text())
        exists = os.path.exists(text)

        color = self.validColor
        message = ""

        if text.strip() == "" and self.must_be_set:
            message = self.required_field_msg
            color = self.errorColor
        elif self.path_format and not re.search("%[0-9]*d", text):
            message = self.path_format_msg
            color = self.errorColor
        elif not exists:
            if self.must_exist and not self.path_format:
                message = self.file_does_not_exist_msg
                color = self.invalidColor


        self.setValidationMessage(message)
        self.pathLine.setToolTip(message)
        palette.setColor(self.pathLine.backgroundRole(), color)

        self.pathLine.setPalette(palette)


    def selectDirectory(self):
        """Pops up the select a directory dialog"""
        self.editing = True
        currentDirectory = self.pathLine.text()

        #if not os.path.exists(currentDirectory):
        #    currentDirectory = "~"

        if self.selectFiles:
            currentDirectory = QtGui.QFileDialog.getOpenFileName(self, "Select a path", currentDirectory)
        else:
            currentDirectory = QtGui.QFileDialog.getExistingDirectory(self, "Select a directory", currentDirectory)

        if not currentDirectory == "":
            self.pathLine.setText(currentDirectory)

        self.editing = False


    def contentsChanged(self):
        """Called whenever the path is changed."""
        if not self.editing:
            self.updateContent(self.pathLine.text())

            
    def fetchContent(self):
        """Retrieves data from the model and inserts it into the edit line"""
        self.editing = True
        
        path = self.getFromModel()
        if path == None:
            path = ""

        self.pathLine.setText(path)
        self.editing = False

