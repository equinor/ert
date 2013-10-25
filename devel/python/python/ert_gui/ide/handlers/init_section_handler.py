import os
from ert_gui.ide.handlers.keyword_handler import KeywordHandler
from ert_gui.ide.keyword import Keyword

class InitSectionHandler(KeywordHandler):
    def __init__(self):
        super(InitSectionHandler, self).__init__("INIT_SECTION")
        self.setExpectedParameterCount(1)


    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() == 1:
            init_section = keyword[0]

            if not os.path.exists(init_section.value):
                init_section.error = True
                init_section.error_message = "File does not exist!"
            elif not os.path.isfile(init_section.value):
                init_section.error = True
                init_section.error_message = "INIT_SECTION expects a file!"

        super(InitSectionHandler, self).handle(keyword, highlighter)