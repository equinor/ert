from PyQt4 import QtGui, QtCore
import matplotlib.figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import datetime
import time
import numpy
from widgets.util import print_timing
from  pages.plot.plotdata import PlotData
import widgets
from matplotlib.dates import AutoDateLocator

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

        #self.canvas.draw = print_timing(self.canvas.draw)

        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlim()

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



        self.errorbar_visible = False
        self.alpha = 0.125
        self.errorbar_limit = 10
        self.xminf = 0.0
        self.xmaxf = 1.0
        self.plot_path = "."

    #@print_timing
    @widgets.util.may_take_a_long_time
    def drawPlot(self):
        self.axes.cla()
        self.lines = []

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

            line, = self.axes.plot_date(x, y, "-", color=self.blue, alpha=self.alpha, picker=2, zorder=100 + member) #list of lines returned (we only add one)
            line.set_gid(member)
            self.lines.append(line)


        if self.errorbar_visible and not self.data.obs_x is None:
            x = self.data.obs_x
            x = [datetime.date(*time.localtime(t)[0:3]) for t in x]
            y = self.data.obs_y
            std = self.data.obs_std

            x = numpy.array(x)
            y = numpy.array(y)
            std = numpy.array(std)

            if len(x) <= self.errorbar_limit:
                self.axes.errorbar(x, y, std, fmt=None, ecolor=self.orange, zorder=10)
            else:
                self.axes.plot_date(x, y, "-", color=self.orange, alpha=0.75) #list of lines returned (we only add one)
                self.axes.plot_date(x, y - std, "--", color=self.orange, alpha=0.75) #list of lines returned (we only add one)
                self.axes.plot_date(x, y + std, "--", color=self.orange, alpha=0.75) #list of lines returned (we only add one)


        self.xlimits = self.axes.get_xlim()
        #self.axes.set_xlim(xlim[0] - 30, xlim[1] + 30)

        #years = matplotlib.dates.YearLocator()   # every year
#        months = matplotlib.dates.MonthLocator()  # every month
#        #yearsFmt = matplotlib.dates.DateFormatter('%b %y')
        yearsFmt = matplotlib.dates.DateFormatter('%b \'%Y')
        #monthFmt = matplotlib.dates.DateFormatter('%b')
        #self.axes.xaxis.set_major_locator(years)
        self.axes.xaxis.set_major_formatter(yearsFmt)
#        self.axes.xaxis.set_minor_locator(months)
        #self.axes.xaxis.set_minor_formatter(monthFmt)
#
#        self.axes.xaxis.set_major_locator(AutoDateLocator())
        #self.axes.xaxis.set_minor_locator(AutoDateLocator())

        self.setXViewFactors(self.xminf, self.xmaxf, False)
        self.fig.autofmt_xdate()
        self.canvas.draw()
        
    def resizeEvent(self, event):
        QtGui.QFrame.resizeEvent(self, event)
        self.canvas.resize(event.size().width(), event.size().height())

    def setData(self, data):
        self.data = data

    def showErrorbar(self, errorbar_visible=False):
        self.errorbar_visible = errorbar_visible
        self.drawPlot()

    def getShowErrorbar(self):
        return self.errorbar_visible

    def setAlphaValue(self, value):
        if value < 0.0:
            self.alpha = 0.0
        elif value > 1.0:
            self.alpha = 1.0
        else:
            self.alpha = value

        for line in self.lines:
            line.set_alpha(self.alpha)

        self.canvas.draw()

    def getAlphaValue(self):
        return self.alpha

    def setErrorbarLimit(self, limit):
        self.errorbar_limit = limit
        self.drawPlot()
    
    def setXViewFactors(self, xminf, xmaxf, draw=True):
        self.xminf = xminf
        self.xmaxf = xmaxf
        x_min = self.convertDate(self.data.x_min)
        x_max = self.convertDate(self.data.x_max)
        range = x_max - x_min
        self.axes.set_xlim(x_min + xminf * range - 60, x_min + xmaxf * range + 60)

        if draw:
            self.canvas.draw()

    def convertDate(self, ert_time):
        if ert_time < 0:
            ert_time = 0
        elif ert_time > time.time():
            ert_time = time.time() + (60 * 60 * 24 * 365 * 10) #ten years into the future
            
        return matplotlib.dates.date2num(datetime.date(*time.localtime(ert_time)[0:3]))

    def save(self):
        path = self.plot_path + "/" + self.axes.get_title()
        self.fig.savefig(path + ".png", dpi=300, format="png")
        self.fig.savefig(path + ".pdf", dpi=300, format="pdf")

    def setPlotPath(self, plot_path):
        self.plot_path = plot_path

