import matplotlib.pyplot as plt

from ert_gui.shell import ShellContext, assertConfigLoaded, matchItems, extractFullArgument, autoCompleteListWithSeparator
from ert_gui.plottery import PlotDataGatherer, PlotConfig, PlotContext
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
        #todo: apply plot settings
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
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)

        cmd_name = "histogram"
        instance.addHelpFunction(cmd_name, "<key_1> [key_2..key_n]", "Plot a histogram for the specified %s key(s)." % name)
        setattr(instance.__class__, "do_%s" % cmd_name, cls.__createDoFunction(ShellPlot.plotHistogram, name))
        setattr(instance.__class__, "complete_%s" % cmd_name, cls.__createCompleteFunction())


    @classmethod
    def addGaussianKDEPlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)

        cmd_name = "density"
        instance.addHelpFunction(cmd_name, "<key_1> [key_2..key_n]", "Plot the GaussianKDE plot for the specified %s key(s)." % name)
        setattr(instance.__class__, "do_%s" % cmd_name, cls.__createDoFunction(ShellPlot.plotGaussianKDE, name))
        setattr(instance.__class__, "complete_%s" % cmd_name, cls.__createCompleteFunction())


    @classmethod
    def addEnsemblePlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)

        cmd_name = "plot"
        instance.addHelpFunction(cmd_name, "<key_1> [key_2..key_n]", "Plot an ensemble plot of the specified %s key(s)." % name)
        setattr(instance.__class__, "do_%s" % cmd_name, cls.__createDoFunction(ShellPlot.plotEnsemble, name))
        setattr(instance.__class__, "complete_%s" % cmd_name, cls.__createCompleteFunction())

    @classmethod
    def addQuantilesPlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)
        cmd_name = "plot_quantile"
        instance.addHelpFunction(cmd_name, "<key_1> [key_2..key_n]", "Plot different statistics for the specified %s key(s)." % name)
        setattr(instance.__class__, "do_%s" % cmd_name, cls.__createDoFunction(ShellPlot.plotQuantiles, name))
        setattr(instance.__class__, "complete_%s" % cmd_name, cls.__createCompleteFunction())

    @classmethod
    def addDistributionPlotSupport(cls, instance, name):
        """
        :type instance: ert_gui.shell.ShellFunction
        """
        cls.__checkForRequiredMethods(instance)
        cmd_name = "distribution"
        instance.addHelpFunction(cmd_name, "<key_1> [key_2..key_n]", "Plot the distribution for the specified %s key(s)." % name)
        setattr(instance.__class__, "do_%s" % cmd_name, cls.__createDoFunction(ShellPlot.plotDistribution, name))
        setattr(instance.__class__, "complete_%s" % cmd_name, cls.__createCompleteFunction())


    @classmethod
    def __createDoPrintFunction(cls, name):
        def do_function(self, line):
            keys = matchItems(line, self.fetchSupportedKeys())

            if len(keys) == 0:
                self.lastCommandFailed("Must have at least one %s key" % name)
                return False

            case_name = self.ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()

            for key in keys:
                pdg = self.plotDataGatherer()
                if pdg.canGatherDataForKey(key):
                    data = pdg.gatherData(self.ert(), case_name, key)
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

        cmd_name = "print"
        instance.addHelpFunction(cmd_name, "<key_1> [key_2..key_n]", "Print the values for the specified %s key(s)." % name)
        setattr(instance.__class__, "do_%s" % cmd_name, cls.__createDoPrintFunction(name))
        setattr(instance.__class__, "complete_%s" % cmd_name, cls.__createCompleteFunction())
