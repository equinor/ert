from ert_gui.ide.keywords import ErtKeywords
from ert_gui.ide.keywords.configuration_line_parser import ConfigurationLineParser
from ert_gui.ide.keywords.data import ConfigurationLine


class ConfigurationLineBuilder(object):

    def __init__(self, keywords):
        super(ConfigurationLineBuilder, self).__init__()

        assert isinstance(keywords, ErtKeywords)
        self.__keywords = keywords
        self.__configuration_line_parser = ConfigurationLineParser()
        self.__configuration_line = None


    def processLine(self, line):
        self.__configuration_line_parser.parseLine(line)
        self.__configuration_line = None

        if self.__configuration_line_parser.hasKeyword():
            keyword = self.__configuration_line_parser.keyword()

            if keyword.value() in self.__keywords:
                keyword_definition = self.__keywords[keyword.value()]
                keyword.setKeywordDefinition(keyword_definition)

            arguments = self.__configuration_line_parser.arguments()
            self.__configuration_line = ConfigurationLine(keyword, arguments)

    def configurationLine(self):
        """ @rtype: ConfigurationLine """
        return self.__configuration_line

    def hasConfigurationLine(self):
        """ @rtype: bool """
        return self.__configuration_line is not None

    def hasComment(self):
        """ @rtype: bool """
        return self.__configuration_line_parser.hasComment()

    def commentIndex(self):
        return self.__configuration_line_parser.commentIndex()


