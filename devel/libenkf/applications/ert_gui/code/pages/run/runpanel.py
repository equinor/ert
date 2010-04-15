from PyQt4 import QtGui, QtCore
import ertwrapper

from widgets.helpedwidget import HelpedWidget, ContentModel
from widgets.util import resourceIcon



class RunPanel(QtGui.QFrame):
    
    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)
        self.setFrameShape(QtGui.QFrame.Panel)
        self.setFrameShadow(QtGui.QFrame.Raised)


        panelLayout = QtGui.QVBoxLayout()
        self.setLayout(panelLayout)

                
        button = QtGui.QPushButton("Refetch")
        self.connect(button, QtCore.SIGNAL('clicked()'), ContentModel.updateObservers)

        panelLayout.addWidget(button)




    def createSeparator(self):
        """Adds a separator line to the panel."""
        qw = QtGui.QWidget()
        qwl = QtGui.QVBoxLayout()
        qw.setLayout(qwl)

        qf = QtGui.QFrame()
        qf.setFrameShape(QtGui.QFrame.HLine)
        qf.setFrameShadow(QtGui.QFrame.Sunken)

        qwl.addSpacing(5)
        qwl.addWidget(qf)
        qwl.addSpacing(5)

        return qw
