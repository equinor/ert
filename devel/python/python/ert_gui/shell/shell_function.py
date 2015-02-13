from ert_gui.shell import createParameterizedHelpFunction, autoCompleteList


class ShellFunction(object):
    command_help_message = "The command: '%s' supports the following keywords:"

    def __init__(self, name, cmd):
        super(ShellFunction, self).__init__()
        self.cmd = cmd
        self.name = name
        """ :type: ErtShell """

        setattr(cmd.__class__, "do_%s" % name, self.doKeywords)
        setattr(cmd.__class__, "complete_%s" % name, self.completeKeywords)
        setattr(cmd.__class__, "help_%s" % name, self.helpKeywords)


    def ert(self):
        """ @rtype: ert.enkf.enkf_main.EnKFMain """
        return self.cmd.ert()

    def isConfigLoaded(self, verbose=True):
        """ @rtype: bool """
        if self.ert() is None:
            if verbose:
                print("Error: A config file has not been loaded!")
            return False
        return True

    def addHelpFunction(self, function_name, parameter, help_message):
        setattr(self.__class__, "help_%s" % function_name, createParameterizedHelpFunction(parameter, help_message))

    def findKeywords(self):
        return [name[3:] for name in dir(self.__class__) if name.startswith("do_")]

    def helpKeywords(self):
        print(self.command_help_message % self.name)
        keywords = self.findKeywords()
        help_format = " %-10s %-25s %s"
        print(help_format % ("Keyword", "Parameter(s)", "Help"))

        for keyword in keywords:
            message = "No help available!"
            parameters = None
            if hasattr(self, "help_%s" % keyword):
                func = getattr(self, "help_%s" % keyword)
                parameters, message = func()

            print(help_format % (keyword, parameters, message))

    def completeKeywords(self, text, line, begidx, endidx):
        line = line[len(self.name) + 1:]
        keyword, argument = self.__getKeywordAndArgument(line)

        if text != keyword and keyword in self.findKeywords(): # the text != keyword is a bit of a hack
            if hasattr(self, "complete_%s" % keyword):
                func = getattr(self, "complete_%s" % keyword)
                return func(text, line, begidx, endidx)
            else:
                return []
        else:
            return autoCompleteList(text, self.findKeywords())

    def doKeywords(self, line):
        keyword, argument = self.__getKeywordAndArgument(line)

        if hasattr(self, "do_%s" % keyword):
            func = getattr(self, "do_%s" % keyword)
            return func(argument)
        else:
            print("Error: Unknown keyword: '%s'" % keyword)


    def __getKeywordAndArgument(self, line):
        line = line.strip()
        argument_index = line.find(" ")

        keyword = line
        argument = ""
        if argument_index > 0:
            keyword = line[:argument_index].strip()
            argument = line[argument_index:].strip()

        return keyword, argument
