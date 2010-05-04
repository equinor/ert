from PyQt4 import QtGui, QtCore
import os
import numpy
from widgets.helpedwidget import ContentModel
import time
import time

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
import ertwrapper

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

        self.setMaximumSize(10000, 1000)

        self.plotData = PlotData()
        self.plotData.setKey("FOPT")


    def resizeEvent(self, event):
        QtGui.QFrame.resizeEvent(self, event)
        self.canvas.resize(event.size().width(), event.size().height())

        self.axes.cla()
        for member in range(0, 50):
            self.axes.plot(self.plotData.data[member]["x_days"], self.plotData.data[member]["y"], "b-")


class PlotData(ContentModel):

    def __init__(self):
        ContentModel.__init__(self)
        self.initialized = False
        self.key = ""


    def initialize(self, ert):
        ert.setTypes("ensemble_config_alloc_keylist")
        ert.setTypes("ensemble_config_get_node", argtypes=ertwrapper.c_char_p)
        ert.setTypes("enkf_main_get_fs")
        ert.setTypes("enkf_fs_has_node", ertwrapper.c_int, argtypes=[ertwrapper.c_long, ertwrapper.c_int, ertwrapper.c_int, ertwrapper.c_int])
        ert.setTypes("enkf_fs_fread_node", None, argtypes=[ertwrapper.c_long, ertwrapper.c_int, ertwrapper.c_int, ertwrapper.c_int])
        ert.setTypes("enkf_node_alloc")
        ert.setTypes("enkf_node_user_get", ertwrapper.c_double, argtypes=[ertwrapper.c_char_p, ertwrapper.c_void_p])
        ert.setTypes("ensemble_config_has_key", ertwrapper.c_int, argtypes=ertwrapper.c_char_p)
        ert.setTypes("enkf_main_get_history_length")
        ert.setTypes("member_config_iget_sim_days", ertwrapper.c_double, argtypes=[ertwrapper.c_int, ertwrapper.c_int])
        ert.setTypes("member_config_iget_sim_time", ertwrapper.c_long, argtypes=[ertwrapper.c_int, ertwrapper.c_int])
        ert.setTypes("enkf_main_get_ensemble_size", ertwrapper.c_int)
        ert.setTypes("enkf_main_iget_member_config", argtypes=ertwrapper.c_int)

        self.initialized = True


    def getter(self, ert):
        results = {}

        ens_conf = ert.ensemble_config
        keys = ert.getStringList(ert.enkf.ensemble_config_alloc_keylist(ens_conf), free_after_use=True)
        results["keys"] = keys

        if ert.enkf.ensemble_config_has_key(ens_conf, self.key):
            config_node = ert.enkf.ensemble_config_get_node(ens_conf, self.key)
            fs = ert.enkf.enkf_main_get_fs(ert.main)
            node = ert.enkf.enkf_node_alloc(config_node)

            stop_time = ert.enkf.enkf_main_get_history_length(ert.main)
            num_realizations = ert.enkf.enkf_main_get_ensemble_size(ert.main)

            for member in range(0, num_realizations):
                results[member] = {}
                results[member]["x_days"] = []
                results[member]["x_time"] = []
                results[member]["y"] = []

                for step in range(1, stop_time + 1):
                    
                    FORECAST = 2
                    if ert.enkf.enkf_fs_has_node(fs, config_node, step, member, FORECAST):

                        member_config = ert.enkf.enkf_main_iget_member_config(ert.main, member)
                        sim_days = ert.enkf.member_config_iget_sim_days(member_config, step, fs)
                        sim_time = ert.enkf.member_config_iget_sim_time(member_config, step, fs)

                        ert.enkf.enkf_fs_fread_node(fs , node , step , member , FORECAST)
                        key_index = None
                        valid = ertwrapper.c_int()
                        value = ert.enkf.enkf_node_user_get(node, key_index, ertwrapper.byref(valid))


                        if valid.value == 1:
                            #print self.key, member, sim_days, time.localtime(sim_time), value
                            results[member]["x_days"].append(sim_days)
                            results[member]["x_time"].append(sim_time)
                            results[member]["y"].append(value)
                            #print self.key, member, sim_days, time.ctime(sim_time), value
                        else:
                            print "Not valid: ", self.key, member, step


        return results

    def fetchContent(self):
        self.data = self.getFromModel()


    def setKey(self, key):
        self.key = key

    def getKeys(self):
        return self.data["keys"]






