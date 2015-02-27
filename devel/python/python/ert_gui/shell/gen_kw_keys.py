from ert.enkf.export.gen_kw_collector import GenKwCollector
from ert_gui.shell import ShellFunction, assertConfigLoaded, extractFullArgument, autoCompleteListWithSeparator, \
    ShellPlot


class GenKWKeys(ShellFunction):
    def __init__(self, cmd):
        super(GenKWKeys, self).__init__("gen_kw", cmd)
        self.addHelpFunction("list", None, "Shows a list of all available gen_kw keys.")
        self.addHelpFunction("histogram", "<key_1> [key_2..key_n]", "Plot the histogram for the specified key(s).")
        self.addHelpFunction("density", "<key_1> [key_2..key_n]", "Plot the density for the specified key(s).")


    def fetchSupportedKeys(self):
        return GenKwCollector.getAllGenKwKeys(self.ert())

    @assertConfigLoaded
    def do_list(self, line):
        keys = sorted(self.fetchSupportedKeys())

        self.columnize(keys)

    @assertConfigLoaded
    def do_histogram(self, line):
        keys = self.splitArguments(line)
        if len(keys) == 0:
            print("Error: Must have at least one GenKW key")
            return False

        for key in keys:
            if key in self.fetchSupportedKeys():
                case_name = self.ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()
                data = GenKwCollector.loadAllGenKwData(self.ert(), case_name, [key])
                ShellPlot.histogram(data, key, log_on_x=key.startswith("LOG10_"))
            else:
                print("Error: Unknown GenKW key '%s'" % key)

    @assertConfigLoaded
    def complete_histogram(self, text, line, begidx, endidx):
        key = extractFullArgument(line, endidx)
        return autoCompleteListWithSeparator(key, self.fetchSupportedKeys())


    @assertConfigLoaded
    def do_density(self, line):
        keys = self.splitArguments(line)
        if len(keys) == 0:
            print("Error: Must have at least one GenKW key")
            return False

        for key in keys:
            if key in self.fetchSupportedKeys():
                case_name = self.ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()
                data = GenKwCollector.loadAllGenKwData(self.ert(), case_name, [key])

                ShellPlot.density(data, key)
            else:
                print("Error: Unknown GenKW key '%s'" % key)

    @assertConfigLoaded
    def complete_density(self, text, line, begidx, endidx):
        key = extractFullArgument(line, endidx)
        return autoCompleteListWithSeparator(key, self.fetchSupportedKeys())
