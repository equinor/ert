from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition, PathArgument, StringArgument, BoolArgument


class ReportKeywords(object):
    def __init__(self, ert_keywords):
        super(ReportKeywords, self).__init__()
        self.group = "Report"

        ert_keywords.addKeyword(self.addReportContext())
        ert_keywords.addKeyword(self.addReportSearchPath())


    def addReportContext(self):
        report_context = ConfigurationLineDefinition(keyword=KeywordDefinition("REPORT_CONTEXT"),
                                                     arguments=[StringArgument(), StringArgument()],
                                                     documentation_link="report/report_context",
                                                     required=False,
                                                     group=self.group)
        return report_context




    def addReportSearchPath(self):
        report_search_path = ConfigurationLineDefinition(keyword=KeywordDefinition("REPORT_SEARCH_PATH"),
                                                     arguments=[PathArgument()],
                                                     documentation_link="report/report_search_path",
                                                     required=False,
                                                     group=self.group)
        return report_search_path
