from ert_gui.shell import ShellFunction, ShellPlot, assertConfigLoaded
from ert_gui.plottery import PlotDataGatherer as PDG


class GenKWKeys(ShellFunction):
    def __init__(self, shell_context):
        super(GenKWKeys, self).__init__("gen_kw", shell_context)
        self.addHelpFunction("list", None, "Shows a list of all available GenKW keys.")

        self.__plot_data_gatherer = None

        ShellPlot.addPrintSupport(self, "GenKW")
        ShellPlot.addHistogramPlotSupport(self, "GenKW")
        ShellPlot.addGaussianKDEPlotSupport(self, "GenKW")
        ShellPlot.addDistributionPlotSupport(self, "GenKW")

    def fetchSupportedKeys(self):
        return self.ert().getKeyManager().genKwKeys()

    def plotDataGatherer(self):
        if self.__plot_data_gatherer is None:
            gen_kw_pdg = PDG.gatherGenKwData
            gen_kw_key_manager = self.ert().getKeyManager().isGenKwKey
            self.__plot_data_gatherer = PDG(gen_kw_pdg, gen_kw_key_manager)

        return self.__plot_data_gatherer

    @assertConfigLoaded
    def do_list(self, line):
        self.columnize(self.fetchSupportedKeys())
