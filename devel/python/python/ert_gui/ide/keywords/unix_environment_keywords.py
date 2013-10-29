from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition, PathArgument, StringArgument, BoolArgument


class UnixEnvironmentKeywords(object):
    def __init__(self, ert_keywords):
        super(UnixEnvironmentKeywords, self).__init__()
        self.group = "Unix"

        ert_keywords.addKeyword(self.addSetEnv())


    def addSetEnv(self):
        setenv = ConfigurationLineDefinition(keyword=KeywordDefinition("SETENV"),
                                             arguments=[StringArgument(), StringArgument(rest_of_line=True,allow_space=True)],
                                             documentation_link="unix_environment/setenv",
                                             required=False,
                                             group=self.group)
        return setenv

