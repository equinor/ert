from ert_gui.plottery import PlotDataGatherer as PDG
from ert_gui.shell import ShellFunction, assertConfigLoaded, ShellPlot

class SummaryKeys(ShellFunction):
    def __init__(self, shell_context):
        super(SummaryKeys, self).__init__("summary", shell_context)

        self.addHelpFunction("list", None, "Shows a list of all available Summary keys. (* = with observations)")
        self.addHelpFunction("observations", None, "Shows a list of all available Summary key observations.")
        self.addHelpFunction("matchers", None, "Shows a list of all Summary keys that the ensemble will match "
                                               "against during simulations and manual load.")
        self.addHelpFunction("add_matcher", "<summary_key>", "Add a matcher to the Summary key matcher set.")

        self.__plot_data_gatherer = None

        ShellPlot.addPrintSupport(self, "Summary")
        ShellPlot.addEnsemblePlotSupport(self, "Summary")
        ShellPlot.addQuantilesPlotSupport(self, "Summary")


    @assertConfigLoaded
    def do_list(self, line):
        keys = self.summaryKeys()
        observation_keys = self.summaryObservationKeys()

        result = ["*%s" % key if key in observation_keys else " %s" % key for key in keys]

        self.columnize(result)

    @assertConfigLoaded
    def do_observations(self, line):
        keys = self.summaryKeys()

        observation_keys = []
        for key in keys:
            obs_keys = self.ert().ensembleConfig().getNode(key).getObservationKeys()
            observation_keys.extend(obs_keys)

        self.columnize(observation_keys)

    @assertConfigLoaded
    def do_matchers(self, line):
        ensemble_config = self.ert().ensembleConfig()
        summary_key_matcher = ensemble_config.getSummaryKeyMatcher()
        keys = sorted(["*%s" % key if summary_key_matcher.isRequired(key) else " %s" % key for key in summary_key_matcher.keys()])

        self.columnize(keys)

    @assertConfigLoaded
    def do_add_matcher(self, line):
        args = self.splitArguments(line)

        if len(args) < 1:
            self.lastCommandFailed("A Summary key is required.")
            return False

        self.ert().ensembleConfig().getSummaryKeyMatcher().addSummaryKey(args[0].strip())


    def summaryKeys(self):
        return self.ert().getKeyManager().summaryKeys()


    def summaryObservationKeys(self):
        return self.ert().getKeyManager().summaryKeysWithObservations()


    def fetchSupportedKeys(self):
        return self.summaryKeys()


    def plotDataGatherer(self):
        if self.__plot_data_gatherer is None:
            summary_pdg = PDG.gatherSummaryData
            summary_key_manager = self.ert().getKeyManager().isSummaryKey
            refcase_pdg = PDG.gatherSummaryRefcaseData
            observation_pdg = PDG.gatherSummaryObservationData
            pdg = PDG(summary_pdg, summary_key_manager, refcaseGatherFunc=refcase_pdg, observationGatherFunc=observation_pdg)
            self.__plot_data_gatherer = pdg

        return self.__plot_data_gatherer
