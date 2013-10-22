import os
from ert_gui.ide.completers.path_completer import PathCompleter
from ert_gui.ide.handlers.keyword_handler import KeywordHandler
from ert_gui.ide.keyword import Keyword

class DataFileHandler(KeywordHandler):
    def __init__(self):
        super(DataFileHandler, self).__init__("DATA_FILE")
        self.setExpectedParameterCount(1)
        self.path_completer = PathCompleter()


    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() == 1:
            data_file = keyword[0]

            if not os.path.exists(data_file.value):
                data_file.error = True
                data_file.error_message = "File does not exist!"
            elif not os.path.isfile(data_file.value):
                data_file.error = True
                data_file.error_message = "DATA_FILE expects a file!"

        super(DataFileHandler, self).handle(keyword, highlighter)

    def parameterOptions(self, keyword, prefix, position_in_block):
        # print(keyword.parameterIndexForPosition(position_in_block), prefix)
        if keyword.parameterIndexForPosition(position_in_block) == 0:
            return self.path_completer.completeOptions(prefix)

        return []

