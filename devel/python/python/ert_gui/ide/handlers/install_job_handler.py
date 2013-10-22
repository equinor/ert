import os
from ert_gui.ide.handlers.keyword_handler import KeywordHandler
from ert_gui.ide.keyword import Keyword

class InstallJobHandler(KeywordHandler):
    def __init__(self):
        super(InstallJobHandler, self).__init__("INSTALL_JOB")
        self.setExpectedParameterCount(2)


    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() == 2:
            job_name = keyword[0]
            job_file = keyword[1]

            if self.NAME_PATTERN.match(job_name.value) is None:
                job_name.error = True
                job_name.error_message = "Parameter is not a proper name. Allowed characters: A-Za-z0-9_"

            if not os.path.exists(job_file.value):
                job_file.error = True
                job_file.error_message = "File does not exist!"
            elif not os.path.isfile(job_file.value):
                job_file.error = True
                job_file.error_message = "DATA_FILE expects a file!"

        super(InstallJobHandler, self).handle(keyword, highlighter)