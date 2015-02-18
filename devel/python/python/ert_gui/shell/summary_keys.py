from ert.enkf import ErtImplType
from ert_gui.shell import ShellFunction, assertConfigLoaded


class SummaryKeys(ShellFunction):
    def __init__(self, cmd):
        super(SummaryKeys, self).__init__("summary", cmd)

        self.addHelpFunction("list", None, "Shows a list of all available summary keys. (* = with observations)")
        self.addHelpFunction("observations", None, "Shows a list of all available summary key observations.")
        self.addHelpFunction("matchers", None, "Shows a list of all summary keys that the ensemble will match "
                                              "against during simulations and manual load.")
        self.addHelpFunction("add_matcher", "<summary_key>", "Add a matcher to the summary key matcher set.")

    @assertConfigLoaded
    def do_list(self, line):
        ensemble_config = self.ert().ensembleConfig()
        keys = sorted([key for key in ensemble_config.getKeylistFromImplType(ErtImplType.SUMMARY)])
        observation_keys = [key for key in keys if len(ensemble_config.getNode(key).getObservationKeys()) > 0]

        result = ["*%s" % key if key in observation_keys else " %s" % key for key in keys]

        self.cmd.columnize(result)

    @assertConfigLoaded
    def do_observations(self, line):
        ensemble_config = self.ert().ensembleConfig()
        keys = sorted([key for key in ensemble_config.getKeylistFromImplType(ErtImplType.SUMMARY)])

        observation_keys = []
        for key in keys:
            obs_keys = ensemble_config.getNode(key).getObservationKeys()
            observation_keys.extend(obs_keys)

        self.cmd.columnize(observation_keys)

    @assertConfigLoaded
    def do_matchers(self, line):
        ensemble_config = self.ert().ensembleConfig()
        summary_key_matcher = ensemble_config.getSummaryKeyMatcher()
        keys = sorted(["*%s" % key if summary_key_matcher.isRequired(key) else " %s" % key for key in summary_key_matcher.keys()])

        self.cmd.columnize(keys)

    @assertConfigLoaded
    def do_add_matcher(self, line):
        args = self.splitArguments(line)

        if len(args) < 1:
            print("Error: A summary key is required.")
            return False


        self.ert().ensembleConfig().getSummaryKeyMatcher().addSummaryKey(args[0].strip())

