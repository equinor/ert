import re
from PyQt4.QtGui import QSyntaxHighlighter, QTextCharFormat, QTextBlockUserData
from ert_gui.ide.keyword import Keyword


class KeywordWrapper(QTextBlockUserData):
    def __init__(self, keyword):
        QTextBlockUserData.__init__(self)
        self.keyword = keyword


class KeywordHandler(object):
    NAME_PATTERN = re.compile("^[A-Za-z0-9_]+$")
    INTEGER_PATTERN = re.compile("^[0-9]+$")

    def __init__(self, keyword_name):
        self.keyword_name = keyword_name
        self.expected_parameter_count = -1

    def setExpectedParameterCount(self, parameter_count):
        self.expected_parameter_count = parameter_count

    def isHandlerFor(self, keyword):
        assert isinstance(keyword, Keyword)
        match = re.match("^%s$" % self.keyword_name, keyword.keyword)
        return match is not None

    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """
        assert isinstance(keyword, Keyword)
        assert isinstance(highlighter, QSyntaxHighlighter)

        highlighter.setCurrentBlockUserData(KeywordWrapper(keyword))
        keyword.handler = self

        if keyword.parameterCount() != self.expected_parameter_count:
            keyword.error = True
            keyword.error_message = "Unexpected number of parameters. Expected %d found %d" % (self.expected_parameter_count, keyword.parameterCount())


        keyword_format = QTextCharFormat(highlighter.keyword_format)

        if keyword.hasError():
            keyword_format.merge(highlighter.error_format)

        highlighter.setFormat(keyword.start, keyword.length, keyword_format)


        for parameter in keyword.parameters():
            if parameter.error:
                highlighter.setFormat(parameter.start, parameter.length, highlighter.error_format)

        if self.expected_parameter_count > 0:

            if keyword.parameterCount() > self.expected_parameter_count:
                for parameter in keyword.parameters()[2:]:
                    highlighter.setFormat(parameter.start, parameter.end, highlighter.error_format)

            elif keyword.parameterCount() < self.expected_parameter_count:
                block_length = len(unicode(highlighter.currentBlock().text()))

                if keyword.parameterCount() > 0:
                    parameter = keyword.parameters()[keyword.parameterCount() - 1]
                    end = parameter.end
                else: # keyword.parameterCount() == 0:
                    end = keyword.end

                length = block_length - end

                if length < 1:
                    length = 1

                highlighter.setFormat(end + 1, length, highlighter.error_format)

    def parameterOptions(self, keyword, prefix, position_in_block):
        return []