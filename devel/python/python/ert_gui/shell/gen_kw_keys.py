from math import ceil, sqrt
from ert.enkf.export.gen_kw_collector import GenKwCollector
from ert_gui.shell import ShellFunction, assertConfigLoaded, autoCompleteList
from ert_gui.shell.shell_tools import extractFullArgument

import pylab
import matplotlib.pyplot as plt

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
    def gen_kw_completer(self, text, line, begidx, endidx):
        key = extractFullArgument(line, endidx)
        if ":" in key:
            text = key
            items = autoCompleteList(text, self.fetchSupportedKeys())
            items = [item[item.rfind(":") + 1:] for item in items if ":" in item]
        else:
            items = autoCompleteList(text, self.fetchSupportedKeys())

        return items

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
                bin_count = int(ceil(sqrt(len(data.index))))
                #todo: missing log binning on the x-axis
                plot = data.plot(kind="hist", alpha=0.75, bins=bin_count)
                plot.set_ylabel("Count")
            else:
                print("Error: Unknown GenKW key '%s'" % key)

    complete_histogram = gen_kw_completer


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
                plot = data.plot(kind="kde", alpha=0.75)
            else:
                print("Error: Unknown GenKW key '%s'" % key)

    complete_density = gen_kw_completer
