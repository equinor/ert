from ert.enkf import ErtImplType
from ert.enkf.export.summary_collector import SummaryCollector
from ert_gui.shell import ShellFunction, assertConfigLoaded, autoCompleteList, extractFullArgument


class SummaryKeys(ShellFunction):
    def __init__(self, cmd):
        super(SummaryKeys, self).__init__("summary", cmd)

        self.addHelpFunction("list", None, "Shows a list of all available summary keys. (* = with observations)")
        self.addHelpFunction("observations", None, "Shows a list of all available summary key observations.")
        self.addHelpFunction("matchers", None, "Shows a list of all summary keys that the ensemble will match "
                                              "against during simulations and manual load.")
        self.addHelpFunction("add_matcher", "<summary_key>", "Add a matcher to the summary key matcher set.")
        self.addHelpFunction("plot", "<key_1> [key_2..key_n]", "Plot the specified key(s).")

    @assertConfigLoaded
    def do_list(self, line):
        ensemble_config = self.ert().ensembleConfig()
        keys = sorted([key for key in ensemble_config.getKeylistFromImplType(ErtImplType.SUMMARY)])
        observation_keys = [key for key in keys if len(ensemble_config.getNode(key).getObservationKeys()) > 0]

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
            print("Error: A summary key is required.")
            return False


        self.ert().ensembleConfig().getSummaryKeyMatcher().addSummaryKey(args[0].strip())


    def summaryKeys(self):
        ensemble_config = self.ert().ensembleConfig()
        return sorted([key for key in ensemble_config.getKeylistFromImplType(ErtImplType.SUMMARY)])


    @assertConfigLoaded
    def do_plot(self, line):
        keys = self.splitArguments(line)
        if len(keys) == 0:
            print("Error: Must have at least one Summary key")
            return False

        for key in keys:
            if key in self.summaryKeys():
                case_name = self.ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()
                data = SummaryCollector.loadAllSummaryData(self.ert(), case_name, [key])
                data.reset_index(inplace=True)
                data = data.pivot(index="Date", columns="Realization", values=key)
                plot = data.plot(alpha=0.75)
            else:
                print("Error: Unknown Summary key '%s'" % key)

    @assertConfigLoaded
    def complete_plot(self, text, line, begidx, endidx):
        key = extractFullArgument(line, endidx)
        if ":" in key:
            text = key
            items = autoCompleteList(text, self.summaryKeys())
            items = [item[item.rfind(":") + 1:] for item in items if ":" in item]
        else:
            items = autoCompleteList(text, self.summaryKeys())

        return items

