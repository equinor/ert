#todo: proper support for unicode characters?

from PyQt4 import QtGui, QtCore
import sys

#for k in QtGui.QStyleFactory.keys():
#    print k
#
#QtGui.QApplication.setStyle("Plastique")
import os
from newconfig import NewConfigurationDialog

app = QtGui.QApplication(sys.argv) #Early so that QT is initialized before other imports

import ertwrapper

from pages.application import Application
from pages.init.initpanel import InitPanel
from pages.run.runpanel import RunPanel
from pages.config.configpages import ConfigPages
from pages.plot.plotpanel import PlotPanel
from widgets.helpedwidget import ContentModel
from widgets.util import resourceImage, resourceIcon

import matplotlib

print "PyQt4 version: ", QtCore.qVersion()
print "matplotlib version: ", matplotlib.__version__

splash = QtGui.QSplashScreen(resourceImage("splash"), QtCore.Qt.WindowStaysOnTopHint)
splash.show()
splash.showMessage("Starting up...", color=QtCore.Qt.white)
app.processEvents()

window = Application()

splash.showMessage("Bootstrapping...", color=QtCore.Qt.white)
app.processEvents()

if os.environ.has_key("ERT_LD_PATH"):
    enkf_so = os.environ["ERT_LD_PATH"]
else:
    sys.exit("Must set environment variable: 'ERT_LD_PATH' to point to directory with ert shared library files.")

ert = ertwrapper.ErtWrapper(enkf_so)

site_config = "/project/res/etc/ERT/Config/site-config"
enkf_config = sys.argv[1]
strict      = True
print "Looking for:%s" % enkf_config
if not os.path.exists(enkf_config):
    print "Trying to start new config"
    new_configuration_dialog = NewConfigurationDialog(enkf_config)
    success = new_configuration_dialog.exec_()
    if not success:
        print "Can not run without a configuration file."
        sys.exit(1)
    else:
        enkf_config      = new_configuration_dialog.getConfigurationPath()
        firste_case_name = new_configuration_dialog.getCaseName()
        dbase_type       = new_configuration_dialog.getDBaseType()
        num_realizations = new_configuration_dialog.getNumberOfRealizations()
        storage_path     = new_configuration_dialog.getStoragePath()
        ert.enkf.enkf_main_create_new_config(enkf_config, storage_path , firste_case_name, dbase_type, num_realizations)
        strict = False

ert.bootstrap(enkf_config, site_config = site_config, strict = strict)
window.setSaveFunction(ert.save)

splash.showMessage("Creating GUI...", color=QtCore.Qt.white)
app.processEvents()

window.addPage("Configuration", resourceIcon("config"), ConfigPages(window))
window.addPage("Init", resourceIcon("db"), InitPanel(window))
window.addPage("Run", resourceIcon("run"), RunPanel(window))
window.addPage("Plots", resourceIcon("plot"), PlotPanel())

splash.showMessage("Communicating with ERT...", color=QtCore.Qt.white)
app.processEvents()

ContentModel.contentModel = ert
ContentModel.updateObservers()

window.show()
splash.finish(window)

sys.exit(app.exec_())




