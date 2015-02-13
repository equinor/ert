from ert_gui.shell import ShellFunction, autoCompleteList, assertConfigLoaded


class Cases(ShellFunction):
    def __init__(self, cmd):
        super(Cases, self).__init__("case", cmd)

        self.addHelpFunction("list", None, "Shows a list of all available cases.")
        self.addHelpFunction("select", "<case_name>", "Change the current file system to the named case.")
        self.addHelpFunction("create", "<case_name>", "Create a new case with the specified named.")

    @assertConfigLoaded
    def do_list(self, line):
        fs_list = self.getFileSystemNames()
        current_fs = self.ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()
        max_length = max([len(fs) for fs in fs_list])
        case_format = "%1s %-" + str(max_length) + "s  %s"
        for fs in fs_list:
            current = ""
            if fs == current_fs:
                current = "*"

            state = "No Data"
            if self.ert().getEnkfFsManager().caseHasData(fs):
                state = "Data"

            print(case_format % (current, fs, state))

    def getFileSystemNames(self):
        return sorted([fs for fs in self.ert().getEnkfFsManager().getCaseList()])


    @assertConfigLoaded
    def do_select(self, case_name):
        case_name = case_name.strip()
        if case_name in self.getFileSystemNames():
            fs = self.ert().getEnkfFsManager().getFileSystem(case_name)
            self.ert().getEnkfFsManager().switchFileSystem(fs)
        else:
            print("Error: Unknown file system '%s'" % case_name)

    @assertConfigLoaded
    def complete_select(self, text, line, begidx, endidx):
        return autoCompleteList(text, self.getFileSystemNames())


    @assertConfigLoaded
    def do_create(self, case_name):
        case_name = case_name.strip()
        if not case_name in self.getFileSystemNames():
            fs = self.ert().getEnkfFsManager().getFileSystem(case_name)
            self.ert().getEnkfFsManager().switchFileSystem(fs)
        else:
            print("Error: Case '%s' already exists!" % case_name)

