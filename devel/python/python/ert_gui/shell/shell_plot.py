import matplotlib.pyplot as plt

from ert_gui.shell import assertConfigLoaded
from ert_gui.plottery import PlotConfig, PlotContext
from ert_gui.shell.libshell import matchItems, extractFullArgument, autoCompleteListWithSeparator
import ert_gui.plottery.plots as plots


class ShellPlot(object):
    @classmethod
    def __createPlotContext(cls, shell_context, data_gatherer, key):
        """
        :type shell_context: ShellContext
        :param data_gatherer: PlotDataGatherer
        :param key: str
        """
        figure = plt.figure()
        figure.set_tight_layout(True)
        cases = shell_context["plot_settings"].getCurrentPlotCases()
        plot_config = PlotConfig(key)
        # todo: apply plot settings
        plot_context = PlotContext(shell_context.ert(), figure, plot_config, cases, key, data_gatherer)
        return plot_context

    @classmethod
    def plotEnsemble(cls, shell_context, data_gatherer, key):
        """
        :type shell_context: ShellContext
        :param data_gatherer: PlotDataGatherer
        :param key: str
        """
        plot_context = cls.__createPlotContext(shell_context, data_gatherer, key)
        plots.plotEnsemble(plot_context)

    @classmethod
    def plotQuantiles(cls, shell_context, data_gatherer, key):
        """
        :type shell_context: ShellContext
        :param data_gatherer: PlotDataGatherer
        :param key: str
        """
        plot_context = cls.__createPlotContext(shell_context, data_gatherer, key)
        plots.plotStatistics(plot_context)

    @classmethod
    def plotHistogram(cls, shell_context, data_gatherer, key):
        """
        :type shell_context: ShellContext
        :param data_gatherer: PlotDataGatherer
        :param key: str
        """
        plot_context = cls.__createPlotContext(shell_context, data_gatherer, key)
        plots.plotHistogram(plot_context)

    @classmethod
    def plotDistribution(cls, shell_context, data_gatherer, key):
        """
        :type shell_context: ShellContext
        :param data_gatherer: PlotDataGatherer
        :param key: str
        """
        plot_context = cls.__createPlotContext(shell_context, data_gatherer, key)
        plots.plotDistribution(plot_context)

    @classmethod
    def plotGaussianKDE(cls, shell_context, data_gatherer, key):
        """
        :type shell_context: ShellContext
        :param data_gatherer: PlotDataGatherer
        :param key: str
        """
        plot_context = cls.__createPlotContext(shell_context, data_gatherer, key)
        plots.plotGaussianKDE(plot_context)

    @classmethod
    def __checkForRequiredMethods(cls, instance):
        if not hasattr(instance, "fetchSupportedKeys"):
            raise NotImplementedError("Class must implement: fetchSupportedKeys()")

        if not hasattr(instance, "plotDataGatherer"):
            raise NotImplementedError("Class must implement: plotDataGatherer()")

    @classmethod
    def __createDoFunction(cls, plot_function, name):
        def do_function(self, line):
            keys = matchItems(line, self.fetchSupportedKeys())

            if len(keys) == 0:
                self.lastCommandFailed("Must have at least one %s key" % name)
                return False

            for key in keys:
                pdg = self.plotDataGatherer()
                plot_function(self.shellContext(), pdg, key)

        return assertConfigLoaded(do_function)

    @classmethod
    def __createCompleteFunction(cls):
        def complete_histogram(self, text, line, begidx, endidx):
            key = extractFullArgument(line, endidx)
            return autoCompleteListWithSeparator(key, self.fetchSupportedKeys())

        complete_histogram = assertConfigLoaded(complete_histogram)
        return complete_histogram

    @classmethod
    def addHistogramPlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.libshell.ShellCollection
        """
        cls.__checkForRequiredMethods(instance)

        instance.addShellFunction(name="histogram",
                                  function=cls.__createDoFunction(ShellPlot.plotHistogram, name),
                                  completer=cls.__createCompleteFunction(),
                                  help_arguments="<key_1> [key_2..key_n]",
                                  help_message="Plot a histogram for the specified %s key(s)." % name)

    @classmethod
    def addGaussianKDEPlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)

        instance.addShellFunction(name="density",
                                  function=cls.__createDoFunction(ShellPlot.plotGaussianKDE, name),
                                  completer=cls.__createCompleteFunction(),
                                  help_arguments="<key_1> [key_2..key_n]",
                                  help_message="Plot a GaussianKDE plot for the specified %s key(s)." % name)

    @classmethod
    def addEnsemblePlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)

        instance.addShellFunction(name="plot",
                                  function=cls.__createDoFunction(ShellPlot.plotEnsemble, name),
                                  completer=cls.__createCompleteFunction(),
                                  help_arguments="<key_1> [key_2..key_n]",
                                  help_message="Plot an ensemble plot for the specified %s key(s)." % name)

    @classmethod
    def addQuantilesPlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)

        instance.addShellFunction(name="plot_quantile",
                                  function=cls.__createDoFunction(ShellPlot.plotQuantiles, name),
                                  completer=cls.__createCompleteFunction(),
                                  help_arguments="<key_1> [key_2..key_n]",
                                  help_message="Plot a different statistics for the specified %s key(s)." % name)

    @classmethod
    def addDistributionPlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)
        instance.addShellFunction(name="distribution",
                                  function=cls.__createDoFunction(ShellPlot.plotDistribution, name),
                                  completer=cls.__createCompleteFunction(),
                                  help_arguments="<key_1> [key_2..key_n]",
                                  help_message="Plot the distribution plot for the specified %s key(s)." % name)

    @classmethod
    def __createDoPrintFunction(cls, name):
        def do_function(self, line):
            keys = matchItems(line, self.fetchSupportedKeys())

            if len(keys) == 0:
                self.lastCommandFailed("Must have at least one %s key" % name)
                return False

            case_name = self.shellContext().ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()

            for key in keys:
                pdg = self.plotDataGatherer()
                if pdg.canGatherDataForKey(key):
                    data = pdg.gatherData(self.shellContext().ert(), case_name, key)
                    print(data)
                else:
                    self.lastCommandFailed("Unable to print data for key: %s" % key)

        return assertConfigLoaded(do_function)

    @classmethod
    def addPrintSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)

        instance.addShellFunction(name="print",
                                  function=cls.__createDoPrintFunction(name),
                                  completer=cls.__createCompleteFunction(),
                                  help_arguments="<key_1> [key_2..key_n]",
                                  help_message="Print the values for the specified %s key(s)." % name)
