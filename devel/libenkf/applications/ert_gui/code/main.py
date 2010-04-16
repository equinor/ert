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

doTheSplash = True

if doTheSplash:
    splash = QtGui.QSplashScreen(widgets.util.resourceImage("splash"))
    splash.show()
    splash.showMessage("Starting up...")
    app.processEvents()
    time.sleep(1)

window = Application()

if doTheSplash:
    splash.showMessage("Bootstrapping...")
    app.processEvents()

site_config = "/project/res/etc/ERT/Config/site-config"
enkf_config = local.enkf_config
enkf_so     = local.enkf_so
ert = ertwrapper.ErtWrapper(site_config = site_config, enkf_config = enkf_config, enkf_so = enkf_so)

if doTheSplash:
    time.sleep(1)


if doTheSplash:
    splash.showMessage("Creating GUI...")
    app.processEvents()
    time.sleep(1)


window.addPage("Configuration", widgets.util.resourceIcon("config"), ConfigPages(window))
window.addPage("Init", widgets.util.resourceIcon("db"), InitPanel(window))
window.addPage("Run", widgets.util.resourceIcon("run"), RunPanel(window))
window.addPage("Plots", widgets.util.resourceIcon("plot"), PlotPanel("plots/default"))


ContentModel.contentModel = ert
ContentModel.updateObservers()


window.show()

if doTheSplash:
    splash.finish(window)

sys.exit(app.exec_())




