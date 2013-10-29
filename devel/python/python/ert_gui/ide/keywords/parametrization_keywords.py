from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition, PathArgument, StringArgument


class ParametrizationKeywords(object):
    def __init__(self, ert_keywords):
        super(ParametrizationKeywords, self).__init__()
        self.group = "Parametrization"

        ert_keywords.addKeyword(self.addField())
        ert_keywords.addKeyword(self.addGenData())
        ert_keywords.addKeyword(self.addGenKw())
        ert_keywords.addKeyword(self.addSummary())



    def addField(self):
        field = ConfigurationLineDefinition(keyword=KeywordDefinition("FIELD"),
                                            arguments=[StringArgument(),StringArgument(), StringArgument(), StringArgument()],
                                            documentation_link="parametrization/field",
                                            required=False,
                                            group=self.group)
        return field



    def addGenData(self):
        gen_data = ConfigurationLineDefinition(keyword=KeywordDefinition("GEN_DATA"),
                                            arguments=[StringArgument(),StringArgument(), StringArgument(), StringArgument()],
                                            documentation_link="parametrization/gen_data",
                                            required=False,
                                            group=self.group)
        return gen_data


    def addGenKw(self):
        gen_kw = ConfigurationLineDefinition(keyword=KeywordDefinition("GEN_KW"),
                                            arguments=[StringArgument(),StringArgument(), StringArgument(), StringArgument()],
                                            documentation_link="parametrization/gen_kw",
                                            required=False,
                                            group=self.group)
        return gen_kw


    def addSummary(self):
        summary = ConfigurationLineDefinition(keyword=KeywordDefinition("SUMMARY"),
                                              arguments=[StringArgument(),StringArgument()],
                                              documentation_link="parametrization/summary",
                                              required=False,
                                              group=self.group)
        return summary