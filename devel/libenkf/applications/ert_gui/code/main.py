# Some comments :)
from PyQt4 import QtGui, QtCore
import sys
import local
import os
import time

import ertwrapper
from pages.application import Application
import widgets.util

#for k in QtGui.QStyleFactory.keys():
#    print k
#
#QtGui.QApplication.setStyle("Plastique")
from pages.plotpanel import PlotPanel
from pages.parameters.parameterpanel import ParameterPanel

#todo: proper support for unicode characters?
from pages.init.initpanel import InitPanel
from pages.run.runpanel import RunPanel
from pages.config.configpages import ConfigPages
from widgets.helpedwidget import ContentModel

app = QtGui.QApplication(sys.argv)


splash = QtGui.QSplashScreen(widgets.util.resourceImage("splash"), QtCore.Qt.WindowStaysOnTopHint)
splash.show()
splash.showMessage("Starting up...", color=QtCore.Qt.white)
app.processEvents()

window = Application()

splash.showMessage("Bootstrapping...", color=QtCore.Qt.white)
app.processEvents()

site_config = "/project/res/etc/ERT/Config/site-config"
enkf_config = local.enkf_config
enkf_so     = local.enkf_so
ert = ertwrapper.ErtWrapper(site_config = site_config, enkf_config = enkf_config, enkf_so = enkf_so)

splash.showMessage("Creating GUI...", color=QtCore.Qt.white)
app.processEvents()

window.addPage("Configuration", widgets.util.resourceIcon("config"), ConfigPages(window))
window.addPage("Init", widgets.util.resourceIcon("db"), InitPanel(window))
window.addPage("Run", widgets.util.resourceIcon("run"), RunPanel(window))
window.addPage("Plots", widgets.util.resourceIcon("plot"), PlotPanel("plots/default"))

splash.showMessage("Communicating with ERT...", color=QtCore.Qt.white)
app.processEvents()

ContentModel.contentModel = ert
ContentModel.updateObservers()


window.show()
splash.finish(window)

sys.exit(app.exec_())




