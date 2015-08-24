from PyQt4.QtCore import Qt
from PyQt4.QtGui import QWidget, QVBoxLayout

from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

class PlotWidget(QWidget):
    def __init__(self, name, short_name, parent=None):
        QWidget.__init__(self, parent)

        self.__name = name
        self.__short_name = short_name
        self.__figure = Figure()
        self.__figure.set_tight_layout(True)
        self.__canvas = FigureCanvas(self.__figure)
        self.__canvas.setParent(self)
        self.__canvas.setFocusPolicy(Qt.StrongFocus)
        self.__canvas.setFocus()

        vbox = QVBoxLayout()
        vbox.addWidget(self.__canvas)
        self.__toolbar = NavigationToolbar(self.__canvas, self)
        vbox.addWidget(self.__toolbar)
        self.setLayout(vbox)


        self.resetPlot()


    def getFigure(self):
        """ :rtype: matplotlib.figure.Figure"""
        return self.__figure


    def resetPlot(self):
        self.__figure.clear()


    def updatePlot(self):
        self.__canvas.draw()
