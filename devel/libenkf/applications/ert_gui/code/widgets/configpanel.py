from PyQt4 import QtGui, QtCore


class ConfigPanel(QtGui.QTabWidget):
    """Convenience class for a tabbed configuration panel"""

    def __init__(self, parent=None):
        """Creates a config panel widget"""
        QtGui.QTabWidget.__init__(self, parent)
        self.layoutQueue = []


    def startPage(self, name):
        """Starts a new page of the configuration panel"""
        self.pageName = name
        self.contentPage = QtGui.QWidget()
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignRight)


    def addRow(self, row):
        """
        Add a new row on a configuration page. Returns the row widget.
        If the row does not have a getLabel() function the row spans both columns.
        """
        if hasattr(row, "getLabel"):
            self.formLayout.addRow(row.getLabel(), row)
        else:
            self.formLayout.addRow(row)
            
        return row

    def endPage(self):
        """Indicate the end of a complete configuration page."""
        self.contentPage.setLayout(self.formLayout)
        self.addTab(self.contentPage, self.pageName)

        self.contentPage = None
        self.formLayout = None
        self.pageName = None

    def startGroup(self, groupTitle):
        """Start a titled sub group on the page."""
        self.groupBox = QtGui.QGroupBox(groupTitle)
        self.layoutQueue.append(self.formLayout)
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignRight)

    def endGroup(self):
        """Finish the titled sub group"""
        self.groupBox.setLayout(self.formLayout)

        self.formLayout = self.layoutQueue.pop()
        self.formLayout.addRow(self.groupBox)
        self.groupBox = None
