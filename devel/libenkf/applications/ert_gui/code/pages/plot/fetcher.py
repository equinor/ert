from PyQt4.QtCore import QObject

class PlotDataFetcherHandler(QObject):

    def __init__(self):
        QObject.__init__(self)

    def isHandlerFor(self, ert, key):
        return False

    def initialize(self, ert):
        pass

    def fetch(self, ert, key, parameter, data):
        pass

    def getConfigurationWidget(self, context_data):
        pass

    def configure(self, parameter, context_data):
        pass