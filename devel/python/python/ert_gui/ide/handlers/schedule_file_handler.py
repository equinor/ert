import os
from ert_gui.ide.handlers.keyword_handler import KeywordHandler
from ert_gui.ide.keyword import Keyword

class ScheduleFileHandler(KeywordHandler):
    def __init__(self):
        super(ScheduleFileHandler, self).__init__("SCHEDULE_FILE")
        self.setExpectedParameterCount(1)


    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() == 1:
            schedule_file = keyword[0]

            if not os.path.exists(schedule_file.value):
                schedule_file.error = True
                schedule_file.error_message = "File does not exist!"
            elif not os.path.isfile(schedule_file.value):
                schedule_file.error = True
                schedule_file.error_message = "SCHEDULE_FILE expects a file!"

        super(ScheduleFileHandler, self).handle(keyword, highlighter)