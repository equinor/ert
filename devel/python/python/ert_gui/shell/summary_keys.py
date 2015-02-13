from ert.enkf import ErtImplType
from ert_gui.shell import ShellFunction, assertConfigLoaded, autoCompleteList


class SummaryKeys(ShellFunction):
    def __init__(self, cmd):
        super(SummaryKeys, self).__init__("summary", cmd)

        self.addHelpFunction("list", None, "Shows a list of all available summary keys.")

    @assertConfigLoaded
    def do_list(self, line):
        keys = [key for key in self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.SUMMARY)]
        self.cmd.columnize(sorted(keys))