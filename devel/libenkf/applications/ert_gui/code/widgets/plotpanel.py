from PyQt4 import QtGui, QtCore
import os

class PlotPanel(QtGui.QWidget):
    def __init__(self, path="plots"):

        self.path = path
        QtGui.QWidget.__init__(self)

        variables = []
        for file in os.listdir(self.path):
            variables.append(file.split(".")[0])

        self.image = QtGui.QPixmap(self.path + variables[0])

        plotLayout = QtGui.QHBoxLayout()

        self.label = QtGui.QLabel()
        self.label.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.label.setFrameShape(QtGui.QFrame.Panel)
        self.label.setFrameShadow(QtGui.QFrame.Sunken)

        plotList = QtGui.QListWidget(self)
        plotList.addItems(variables)
        plotList.sortItems()
        plotList.setMaximumWidth(150)
        plotList.setMinimumWidth(150)

        self.connect(plotList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem *, QListWidgetItem *)'), self.select)
        plotLayout.addWidget(plotList)

        self.label.resizeEvent = self.resizeImage
        self.label.sizeHint = lambda : QtCore.QSize(0, 0)
        self.label.minimumSizeHint = lambda : QtCore.QSize(0, 0)

        plotLayout.addWidget(self.label)

        self.setLayout(plotLayout)



    def resizeImage(self, resizeEvent):
        self.scaleImage(resizeEvent.size())


    def select(self, current, previous):
        self.image = QtGui.QPixmap(self.path + "/" + str(current.text()))
        self.scaleImage(self.label.size())


    def scaleImage(self, size):
        self.label.setPixmap(self.image.scaled(size.width() - 2, size.height() - 2, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))




