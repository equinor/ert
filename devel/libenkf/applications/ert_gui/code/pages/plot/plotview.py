from PyQt4 import QtGui, QtCore
import matplotlib.figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import datetime
import time
import numpy
from widgets.util import print_timing
from  pages.plot.plotdata import PlotData

class PlotView(QtGui.QFrame):
    """PlotPanel shows available plot result files and displays them"""
#    blue = (56/255.0, 108/255.0, 176/255.0)
#    yellow = (255/255.0, 255/255.0, 153/255.0)
#    orange = (253/255.0, 192/255.0, 134/255.0)
#    purple = (190/255.0, 174/255.0, 212/255.0)
#    green = (127/255.0, 201/255.0, 127/255.0)

    red = (255/255.0, 26/255.0, 28/255.0)
    blue = (55/255.0, 126/255.0, 184/255.0)
    green = (77/255.0, 175/255.0, 74/255.0)
    purple = (152/255.0, 78/255.0, 163/255.0)
    orange = (255/255.0, 127/255.0, 0/255.0)


    def __init__(self):
        """Create a PlotPanel"""
        QtGui.QFrame.__init__(self)

        self.data = PlotData()

        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.fig = matplotlib.figure.Figure(dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.axes = self.fig.add_subplot(111)



        def onclick(event):
            print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
                event.button, event.x, event.y, event.xdata, event.ydata)
        self.fig.canvas.mpl_connect('button_press_event', onclick)

        def onpick(event):
            thisline = event.artist
            thisline.set_color(self.purple)
            thisline.set_zorder(1000)
            thisline.set_alpha(0.5)
            xdata, ydata = thisline.get_data()
            ind = event.ind
            print 'on pick line:', zip(xdata[ind], ydata[ind]) , thisline.get_gid()
            self.canvas.draw()


        self.fig.canvas.mpl_connect('pick_event', onpick)

    #@print_timing
    def drawPlot(self):
        self.axes.cla()

        name = self.data.getName()
        key_index = self.data.getKeyIndex()

        if not key_index is None:
            name = "%s (%s)" % (name, key_index)

        self.axes.set_title(name)


        for member in self.data.x_data.keys():
            x = self.data.x_data[member]
            x = [datetime.date(*time.localtime(t)[0:3]) for t in x]
            y = self.data.y_data[member]


            x = numpy.array(x)
            y = numpy.array(y)

            #line, = self.axes.plot_date(x, y, "b-", color=self.blue, alpha=0.075, picker=2, zorder=100 + member) #list of lines returned (we only add one)
            line, = self.axes.plot_date(x, y, "b-", color=self.blue, alpha=0.125, picker=2, zorder=100 + member) #list of lines returned (we only add one)
            line.set_gid(member)


        if not self.data.obs_x is None:
            x = self.data.obs_x
            x = [datetime.date(*time.localtime(t)[0:3]) for t in x]
            y = self.data.obs_y
            std = self.data.obs_std

            x = numpy.array(x)
            y = numpy.array(y)
            std = numpy.array(std)
            #line, = self.axes.plot_date(x, y, "yo", picker=2) #list of lines returned (we only add one)
            #self.axes.errorbar(x, y, std, fmt=None, ecolor=(0.0, 0.0, 0.0), barsabove=True, zorder=1000) #list of lines returned (we only add one)
            #self.axes.errorbar(x, y, std, fmt=None, ecolor=self.orange, zorder=10) #list of lines returned (we only add one)
            #self.axes.plot_date(x, y, "-", color=(1.0, 0.0, 0.0), alpha=0.75) #list of lines returned (we only add one)
            #self.axes.plot_date(x, y - std, "--", color=(1.0, 0.0, 0.0), alpha=0.75) #list of lines returned (we only add one)
            #self.axes.plot_date(x, y + std, "--", color=(1.0, 0.0, 0.0), alpha=0.75) #list of lines returned (we only add one)

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

    def setData(self, data):
        self.data = data
    




