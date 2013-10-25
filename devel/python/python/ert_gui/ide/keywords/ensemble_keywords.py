from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition


class EnsembleKeywords(object):
    def __init__(self, ert_keywords):
        super(EnsembleKeywords, self).__init__()
        group = "Ensemble"

        num_realizations = ConfigurationLineDefinition(keyword=KeywordDefinition("NUM_REALIZATIONS"),
                                                       arguments=[IntegerArgument(from_value=1)],
                                                       documentation_link="ensemble/num_realizations",
                                                       required=True,
                                                       group=group)

        ert_keywords.addKeyword(num_realizations)


