from PyQt4 import QtGui, QtCore
import os
import numpy

class ImagePlotPanel(QtGui.QFrame):
    """PlotPanel shows available plot result files and displays them"""
    def __init__(self, path="plots"):
        """Create a PlotPanel"""

        self.path = path
        QtGui.QFrame.__init__(self)

        imageFile = None
        files = []
        if os.path.exists(self.path):
            for file in os.listdir(self.path):
                files.append(file.split(".")[0])
            imageFile = self.path + files[0]


        self.image = QtGui.QPixmap(imageFile)

        plotLayout = QtGui.QHBoxLayout()

        self.label = QtGui.QLabel()
        self.label.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.label.setFrameShape(QtGui.QFrame.StyledPanel)
        #self.label.setFrameShadow(QtGui.QFrame.Sunken)

        plotList = QtGui.QListWidget(self)
        plotList.addItems(files)
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

        self.setFrameShape(QtGui.QFrame.Panel)
        self.setFrameShadow(QtGui.QFrame.Raised)


        # thumbnails -> slow loading of page
        #plotList.setViewMode(QtGui.QListView.IconMode)
        #plotList.setIconSize(QtCore.QSize(96, 96))
        #self.contentsWidget.setMovement(QtGui.QListView.Static)
        #for index in range(plotList.count()):
        #    item = plotList.item(index)
        #    icon = QtGui.QIcon(self.path + "/" + str(item.text()))
        #    item.setIcon(icon)
        #    item.setTextAlignment(QtCore.Qt.AlignHCenter)
        #    item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)



    def resizeImage(self, resizeEvent):
        """Rescale image when panel is resized"""
        self.scaleImage(resizeEvent.size())


    def select(self, current, previous):
        """Update the image current representation by selecting from the list"""
        self.image = QtGui.QPixmap(self.path + "/" + str(current.text()))
        self.scaleImage(self.label.size())


    def scaleImage(self, size):
        """Scale and update the displayed image"""
        if not self.image.isNull():
            self.label.setPixmap(self.image.scaled(size.width(), size.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))



import matplotlib.figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

class PlotPanel(QtGui.QFrame):
    """PlotPanel shows available plot result files and displays them"""
    def __init__(self):
        """Create a PlotPanel"""
        QtGui.QFrame.__init__(self)

        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.fig = matplotlib.figure.Figure(dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        rect = 0.07, 0.07, 0.88, 0.88
        self.axes = self.fig.add_axes(rect)


        x = numpy.arange(0.0, 2.0, 0.02)
        y = numpy.sin(2 * numpy.pi * x)
        l = self.axes.plot(x, y, 'r-')

        self.axes.set_ylim(numpy.min(y) - 0.5, numpy.max(y) + 0.5)

        self.setMaximumSize(10000, 10000)


    def resizeEvent(self, event):
        QtGui.QFrame.resizeEvent(self, event)
        self.canvas.resize(event.size().width(), event.size().height())

