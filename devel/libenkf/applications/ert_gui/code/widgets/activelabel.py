
from PyQt4 import QtGui, QtCore
from widgets.helpedwidget import HelpedWidget

class ActiveLabel(HelpedWidget):
    """Label shows a string. The data structure expected from the getter is a string."""

    def __init__(self, parent=None, label="", help="", default_string=""):
        """Construct a StringBox widget"""
        HelpedWidget.__init__(self, parent, label, help)

        self.active_label = QtGui.QLabel()
        self.addWidget(self.active_label)

        font = self.active_label.font()
        font.setWeight(QtGui.QFont.Bold)
        self.active_label.setFont(font)

        #self.addHelpButton()

        self.active_label.setText(default_string)


    def fetchContent(self):
        """Retrieves data from the model and inserts it into the edit line"""
        self_get_from_model = self.getFromModel()
        if self_get_from_model is None:
            self_get_from_model = ""

        self.active_label.setText(self_get_from_model)
