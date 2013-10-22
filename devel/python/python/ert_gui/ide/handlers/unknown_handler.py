from ert_gui.ide.handlers.keyword_handler import KeywordHandler, KeywordWrapper


class UnknownHandler(KeywordHandler):
    def __init__(self):
        super(UnknownHandler, self).__init__("[A-Z_]+")

    def handle(self, keyword, highlighter):
        keyword.error = True
        keyword.error_message = "Unknown keyword"

        highlighter.setCurrentBlockUserData(KeywordWrapper(keyword))
        highlighter.setFormat(keyword.start, keyword.length, highlighter.error_format)