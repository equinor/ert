from PyQt4 import QtGui, QtCore
import matplotlib.figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from pages.plot.plotdata import PlotContextDataFetcher, PlotDataFetcher
import datetime
import time
import numpy
from widgets.util import print_timing

class PlotView(QtGui.QFrame):
    """PlotPanel shows available plot result files and displays them"""
    def __init__(self):
        """Create a PlotPanel"""
        QtGui.QFrame.__init__(self)

        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.fig = matplotlib.figure.Figure(dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.axes = self.fig.add_subplot(111)

        self.plotDataFetcher = PlotDataFetcher()
        self.plotContextDataFetcher = PlotContextDataFetcher()

        def onclick(event):
            print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
                event.button, event.x, event.y, event.xdata, event.ydata)
        self.fig.canvas.mpl_connect('button_press_event', onclick)

        def onpick(event):
            thisline = event.artist
            thisline.set_color("r")
            xdata, ydata = thisline.get_data()
            ind = event.ind
            print 'on pick line:', zip(xdata[ind], ydata[ind]) , thisline.get_gid()
            self.canvas.draw()


        self.fig.canvas.mpl_connect('pick_event', onpick)

    #@print_timing
    def drawPlot(self):
        self.axes.cla()

        key = self.plotDataFetcher.parameter.getName()
        self.axes.set_title(key)
        for member in range(0, 50):
            x = self.plotDataFetcher.data.x_data[member]
            x = [datetime.date(*time.localtime(t)[0:3]) for t in x]
            y = self.plotDataFetcher.data.y_data[member]

            x = numpy.array(x)
            y = numpy.array(y)
            line, = self.axes.plot_date(x, y, "b-", picker=2) #list of lines returned (we only add one)
            line.set_gid(member)


        years = matplotlib.dates.YearLocator()   # every year
        months = matplotlib.dates.MonthLocator()  # every month
        #yearsFmt = matplotlib.dates.DateFormatter('%b %y')
        yearsFmt = matplotlib.dates.DateFormatter('%Y')
        self.axes.xaxis.set_major_locator(years)
        self.axes.xaxis.set_major_formatter(yearsFmt)
        self.axes.xaxis.set_minor_locator(months)

        self.canvas.draw()


    def resizeEvent(self, event):
        QtGui.QFrame.resizeEvent(self, event)
        self.canvas.resize(event.size().width(), event.size().height())




