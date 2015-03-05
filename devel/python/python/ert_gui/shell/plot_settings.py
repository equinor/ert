from ert_gui.shell import ShellFunction, assertConfigLoaded, autoCompleteList


class PlotSettings(ShellFunction):

    def __init__(self, shell_context):
        super(PlotSettings, self).__init__("plot_settings", shell_context)

        self.__cases = None

        self.addHelpFunction("current", None, "Shows the selected plot source cases.")
        self.addHelpFunction("select", "[case_1..case_n]", "Select one or more cases as default plot sources. "
                                                           "Empty resets to current case.")

        shell_context["plot_settings"] = self

    def getCurrentPlotCases(self):
        """ @rtype: list of str """

        if self.__cases is None:
            case_name = self.ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()
            self.__cases = [case_name]

        return self.__cases

    @assertConfigLoaded
    def do_current(self, line):
        keys = sorted(self.getCurrentPlotCases())
        self.columnize(keys)


    @assertConfigLoaded
    def do_select(self, line):
        case_names = self.splitArguments(line)

        possible_cases = self.getAllCaseList()
        cases = []
        for case_name in case_names:
            if case_name in possible_cases:
                cases.append(case_name)
            else:
                print("Error: Unknown case '%s'" % case_name)

        if len(cases) > 0:
            self.__cases = cases
        else:
            self.__cases = None


    @assertConfigLoaded
    def complete_select(self, text, line, begidx, endidx):
        return autoCompleteList(text, self.getAllCaseList())


    def getAllCaseList(self):
        fs_manager = self.ert().getEnkfFsManager()
        all_case_list = fs_manager.getCaseList()
        all_case_list = [case for case in all_case_list if fs_manager.caseHasData(case)]
        return all_case_list