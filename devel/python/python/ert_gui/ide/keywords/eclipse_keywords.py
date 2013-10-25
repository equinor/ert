from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition
from ert_gui.ide.keywords.definitions.path_argument import PathArgument


class EclipseKeywords(object):
    def __init__(self, ert_keywords):
        super(EclipseKeywords, self).__init__()
        group = "Eclipse"

        data_file = ConfigurationLineDefinition(keyword=KeywordDefinition("DATA_FILE"),
                                                       arguments=[PathArgument()],
                                                       documentation_link="eclipse/data_file",
                                                       required=True,
                                                       group=group)

        ert_keywords.addKeyword(data_file)


