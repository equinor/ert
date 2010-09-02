from PyQt4.QtCore import QObject

class PlotDataFetcherHandler(QObject):
    """An interface for data fetchers."""

    def __init__(self):
        QObject.__init__(self)

    def isHandlerFor(self, ert, key):
        """Retrun True if this handler can handle key."""
        return False

    def initialize(self, ert):
        """Prototype functions and types."""
        pass

    def fetch(self, ert, key, parameter, data, comparison_fs):
        """
        Fetch data from ert by key. Key has already been tested with isHandlerFor()
        Parameter contains configuration data. data is the target.
        comparison_fs is the fs to the case which the plotter should use for comparison plot data
        """
        pass

    def getConfigurationWidget(self, context_data):
        """Return a widget to configure this handler."""
        pass

    def configure(self, parameter, context_data):
        """
        Set the current parameter to configure.
        Should always be called before getConfigurationWidget().
        """
        pass