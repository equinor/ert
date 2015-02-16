from ert.enkf.plot.ensemble_gen_kw_fetcher import EnsembleGenKWFetcher
from ert_gui.shell import ShellFunction, assertConfigLoaded


class GenKWKeys(ShellFunction):
    def __init__(self, cmd):
        super(GenKWKeys, self).__init__("gen_kw", cmd)
        self.addHelpFunction("list", None, "Shows a list of all available gen_kw keys.")

    @assertConfigLoaded
    def do_list(self, line):
        fetcher = EnsembleGenKWFetcher(self.ert())
        keys = sorted(fetcher.fetchSupportedKeys())

        self.cmd.columnize(keys)

