from PyQt4 import QtGui, QtCore
from widgets.helpedwidget import HelpedWidget

class ReloadButton(HelpedWidget):
    """Presents a reload button. """

    def __init__(self, parent=None, label="", help="", button_text=""):
        """Construct a StringBox widget"""
        HelpedWidget.__init__(self, parent, label, help)

        self.button = QtGui.QToolButton(self)
        self.button.setText(button_text)
        self.addWidget(self.button)

        self.connect(self.button, QtCore.SIGNAL('clicked()'), self.fetchContent)

        self.addStretch()
        self.addHelpButton()


    def fetchContent(self):
        """Retrieves data from the model"""
        data = self.getFromModel()
