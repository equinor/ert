from ert_gui.ide.handlers.keyword_handler import KeywordHandler


class DefineHandler(KeywordHandler):
    def __init__(self):
        super(DefineHandler, self).__init__("DEFINE")
        self.setExpectedParameterCount(2)


    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() == 2:
            key = keyword[0]
            value = keyword[1]

            if self.NAME_PATTERN.match(key.value) is None:
                key.error = True
                key.error_message = "Parameter is not a proper name. Allowed characters: A-Za-z0-9_"

        super(DefineHandler, self).handle(keyword, highlighter)