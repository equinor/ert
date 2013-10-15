from ert_gui.ide.handlers.keyword_handler import KeywordHandler


class NumRealizationsHandler(KeywordHandler):
    def __init__(self):
        super(NumRealizationsHandler, self).__init__("NUM_REALIZATIONS")
        self.setExpectedParameterCount(1)

    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() == 1:
            p1 = keyword[0]

            if self.INTEGER_PATTERN.match(p1.value) is None:
                p1.error = True
                p1.error_message = "Parameter should be a positive integer."

        super(NumRealizationsHandler, self).handle(keyword, highlighter)
