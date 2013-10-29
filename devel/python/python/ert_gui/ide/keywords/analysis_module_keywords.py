from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition, PathArgument, StringArgument


class AnalysisModuleKeywords(object):
    def __init__(self, ert_keywords):
        super(AnalysisModuleKeywords, self).__init__()
        self.group = "Analysis Module"

        ert_keywords.addKeyword(self.addAnalysisLoad())
        ert_keywords.addKeyword(self.addAnalysisSelect())
        ert_keywords.addKeyword(self.addAnalysisCopy())
        ert_keywords.addKeyword(self.addAnalysisSetVar())



    def addAnalysisLoad(self):
        analysis_load = ConfigurationLineDefinition(keyword=KeywordDefinition("ANALYSIS_LOAD"),
                                                    arguments=[StringArgument(),PathArgument()],
                                                    documentation_link="analysis_module/analysis_load",
                                                    required=False,
                                                    group=self.group)
        return analysis_load



    def addAnalysisSelect(self):
        analysis_select = ConfigurationLineDefinition(keyword=KeywordDefinition("ANALYSIS_SELECT"),
                                                      arguments=[StringArgument()],
                                                      documentation_link="analysis_module/analysis_select",
                                                      required=False,
                                                      group=self.group)
        return analysis_select


    def addAnalysisSetVar(self):
        analysis_set_var = ConfigurationLineDefinition(keyword=KeywordDefinition("ANALYSIS_SET_VAR"),
                                                       arguments=[StringArgument(),
                                                                  StringArgument(),
                                                                  StringArgument(rest_of_line=True,allow_space=True)],
                                                       documentation_link="analysis_module/analysis_set_var",
                                                       required=False,
                                                       group=self.group)
        return analysis_set_var



    def addAnalysisCopy(self):
        analysis_copy = ConfigurationLineDefinition(keyword=KeywordDefinition("ANALYSIS_COPY"),
                                                    arguments=[StringArgument(), StringArgument()],
                                                    documentation_link="analysis_module/analysis_copy",
                                                    required=False,
                                                    group=self.group)
        return analysis_copy
