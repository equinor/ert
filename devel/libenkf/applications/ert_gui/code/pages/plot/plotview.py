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
from PyQt4.QtCore import SIGNAL
import os

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

        self.selected_lines = []

        def onclick(event):
            #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(evnt.button, evnt.x, evnt.y, evnt.xdata, evnt.ydata)
            pass
        self.fig.canvas.mpl_connect('button_press_event', onclick)

        def onpick(event):
            if event.mouseevent.button == 1:
                thisline = event.artist
                self.toggleLine(thisline)

        self.fig.canvas.mpl_connect('pick_event', onpick)

        def motion_notify_event(event):
            if not self.data is None and not event.xdata is None and not event.ydata is None:
                if self.data.getXDataType() == "time":
                    date = matplotlib.dates.num2date(event.xdata)
                    self.setToolTip("x: %s y: %04f" % (date.strftime("%d/%m-%Y"), event.ydata))
                else:
                    self.setToolTip("x: %04f y: %04f" % (event.xdata, event.ydata))
            else:
                self.setToolTip("")

        self.fig.canvas.mpl_connect('motion_notify_event', motion_notify_event)


        self.errorbar_visible = False
        self.alpha = 0.125
        self.errorbar_limit = 10
        self.xminf = 0.0
        self.xmaxf = 1.0
        self.plot_path = "."
        self.show_stdv = True

        self.plot_type = "-"
        self.observation_plot_type = "-"


    def toggleLine(self, thisline):
        if thisline in self.selected_lines:
            self.clearLine(thisline)
        else:
            thisline.set_color(self.purple)
            thisline.set_zorder(9)
            thisline.set_alpha(0.5)
            self.selected_lines.append(thisline)
        self.emit(SIGNAL('plotSelectionChanged(array)'), self.selected_lines)
        self.canvas.draw()

    def clearLine(self, line):
        line.set_color(self.blue)
        line.set_alpha(self.alpha)
        line.set_zorder(100 + line.get_gid())
        self.selected_lines.remove(line)

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

        if self.data.hasInvertedYAxis() and not self.axes.yaxis_inverted():
            self.axes.invert_yaxis()
        elif not self.data.hasInvertedYAxis() and self.axes.yaxis_inverted():
            self.axes.invert_yaxis()
                

        selected_members = []

        for selected_line in self.selected_lines:
            selected_members.append(selected_line.get_gid())

        self.selected_lines = []

        for member in self.data.x_data.keys():
            x = self.data.x_data[member]
            if self.data.getXDataType() == "time":
                x = [datetime.date(*time.localtime(t)[0:3]) for t in x]

            y = self.data.y_data[member]

            x = numpy.array(x)
            y = numpy.array(y)

            if member in selected_members:
                if self.data.getXDataType() == "time":
                    line, = self.axes.plot_date(x, y, self.plot_type, color=self.purple, alpha=0.5, picker=2, zorder=1) #list of lines returned (we only add one)
                else:
                    line, = self.axes.plot(x, y, self.plot_type, color=self.purple, alpha=0.5, picker=2, zorder=1) #list of lines returned (we only add one)
                self.selected_lines.append(line)
            else:
                if self.data.getXDataType() == "time":
                    line, = self.axes.plot_date(x, y, self.plot_type, color=self.blue, alpha=self.alpha, picker=2, zorder=1) #list of lines returned (we only add one)
                else:
                    line, = self.axes.plot(x, y, self.plot_type, color=self.blue, alpha=self.alpha, picker=2, zorder=1) #list of lines returned (we only add one)
            line.set_gid(member)
            self.lines.append(line)


        if self.errorbar_visible and not (self.data.obs_x is None and self.data.obs_y is None):
            x = self.data.obs_x
            if self.data.getXDataType() == "time":
                x = [datetime.date(*time.localtime(t)[0:3]) for t in x]
                
            y = self.data.obs_y

            x_std = None
            y_std = None

            if not self.data.obs_std_x is None:
                x_std = self.data.obs_std_x
                x_std = numpy.array(x_std)

            if not self.data.obs_std_y is None:
                y_std = self.data.obs_std_y
                y_std = numpy.array(y_std)

            x = numpy.array(x)
            y = numpy.array(y)


            if len(x) <= self.errorbar_limit:
                self.axes.errorbar(x, y, yerr = y_std, xerr = x_std, fmt=None, ecolor=self.orange, zorder=10)
            else:
                if self.data.getXDataType() == "time":
                    self.axes.plot_date(x, y, self.observation_plot_type, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)
                else:
                    self.axes.plot(x, y, self.observation_plot_type, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)

                if self.show_stdv:
                    eblt = ":" #error_bar_line_type
                    if not y_std is None:
                        if self.data.getXDataType() == "time":
                            self.axes.plot_date(x, y - y_std, eblt, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)
                            self.axes.plot_date(x, y + y_std, eblt, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)
                        else:
                            self.axes.plot(x, y - y_std, eblt, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)
                            self.axes.plot(x, y + y_std, eblt, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)

                    elif not x_std is None:
                        if self.data.getXDataType() == "time":
                            self.axes.plot_date(x, y - x_std, eblt, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)
                            self.axes.plot_date(x, y + x_std, eblt, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)
                        else:
                            self.axes.plot(x, y - x_std, eblt, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)
                            self.axes.plot(x, y + x_std, eblt, color=self.orange, alpha=0.75, zorder=10) #list of lines returned (we only add one)



        self.xlimits = self.axes.get_xlim()
        #self.axes.set_xlim(xlim[0] - 30, xlim[1] + 30)

        if self.data.getXDataType() == "time":
            #years = matplotlib.dates.YearLocator()   # every year
            #months = matplotlib.dates.MonthLocator()  # every month
            #yearsFmt = matplotlib.dates.DateFormatter('%b %y')
            yearsFmt = matplotlib.dates.DateFormatter('%b \'%Y')
            #monthFmt = matplotlib.dates.DateFormatter('%b')
            #self.axes.xaxis.set_major_locator(years)
            self.axes.xaxis.set_major_formatter(yearsFmt)
            self.fig.autofmt_xdate()
            #self.axes.xaxis.set_minor_locator(months)
            #self.axes.xaxis.set_minor_formatter(monthFmt)
            #self.axes.xaxis.set_major_locator(AutoDateLocator())
            #self.axes.xaxis.set_minor_locator(AutoDateLocator())

        #number_formatter = matplotlib.ticker.FormatStrFormatter("%f")
        number_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        number_formatter.set_scientific(True)
        #number_formatter.set_powerlimits((5, -5))

        self.axes.yaxis.set_major_formatter(number_formatter)
        self.setXViewFactors(self.xminf, self.xmaxf, False)

        self.canvas.draw()
        
    def resizeEvent(self, event):
        QtGui.QFrame.resizeEvent(self, event)
        self.canvas.resize(event.size().width(), event.size().height())

    def setData(self, data):
        self.data = data

    def showErrorbar(self, errorbar_visible=False, redraw=True):
        self.errorbar_visible = errorbar_visible

        if redraw:
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

    def setShowSTDV(self, show_stdv):
        self.show_stdv = show_stdv
        self.drawPlot()
    
    def setXViewFactors(self, xminf, xmaxf, draw=True):
        self.xminf = xminf
        self.xmaxf = xmaxf

        if self.data.getXDataType() == "time":
            x_min = self.convertDate(self.data.x_min)
            x_max = self.convertDate(self.data.x_max)
            range = x_max - x_min
            self.axes.set_xlim(x_min + xminf * range - 60, x_min + xmaxf * range + 60)
        else:
            x_min = self.data.x_min
            x_max = self.data.x_max
            range = x_max - x_min
            self.axes.set_xlim(x_min + xminf * range - range*0.05, x_min + xmaxf * range + range*0.05)

        if draw:
            self.canvas.draw()

    def convertDate(self, ert_time):
        if ert_time < 0:
            ert_time = 0
        elif ert_time > time.time():
            ert_time = time.time() + (60 * 60 * 24 * 365 * 10) #ten years into the future
            
        return matplotlib.dates.date2num(datetime.date(*time.localtime(ert_time)[0:3]))

    def save(self):
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
            
        path = self.plot_path + "/" + self.axes.get_title()
        self.fig.savefig(path + ".png", dpi=300, format="png")
        self.fig.savefig(path + ".pdf", dpi=300, format="pdf")

    def setPlotPath(self, plot_path):
        self.plot_path = plot_path

    def clearSelection(self):
        lines = [line for line in self.selected_lines]
        for line in lines:
            self.clearLine(line)

        self.emit(SIGNAL('plotSelectionChanged(array)'), self.selected_lines)
        self.canvas.draw()

    def setPlotType(self, plot_type):
        self.plot_type = str(plot_type)
        self.drawPlot()

    def setObservationPlotType(self, plot_type):
        self.observation_plot_type = str(plot_type)
        self.drawPlot()