import os
from ert_gui.ide.handlers.keyword_handler import KeywordHandler
from ert_gui.ide.keyword import Keyword

class GridHandler(KeywordHandler):
    def __init__(self):
        super(GridHandler, self).__init__("GRID")
        self.setExpectedParameterCount(1)


    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() == 1:
            grid_file = keyword[0]

            if not os.path.exists(grid_file.value):
                grid_file.error = True
                grid_file.error_message = "File does not exist!"
            elif not os.path.isfile(grid_file.value):
                grid_file.error = True
                grid_file.error_message = "SCHEDULE_FILE expects a file!"

        super(GridHandler, self).handle(keyword, highlighter)