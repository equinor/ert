from ert.enkf import ErtImplType
from ert_gui.shell import ShellFunction, assertConfigLoaded


class SummaryKeys(ShellFunction):
    def __init__(self, cmd):
        super(SummaryKeys, self).__init__("summary", cmd)

        self.addHelpFunction("list", None, "Shows a list of all available summary keys. (* = with observations)")
        self.addHelpFunction("observations", None, "Shows a list of all available summary key observations.")

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

