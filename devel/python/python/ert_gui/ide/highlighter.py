import re
from PyQt4.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from ert_gui.ide.handlers.data_file_handler import DataFileHandler
from ert_gui.ide.handlers.define_handler import DefineHandler
from ert_gui.ide.handlers.grid_handler import GridHandler
from ert_gui.ide.handlers.init_section_handler import InitSectionHandler
from ert_gui.ide.handlers.install_job_handler import InstallJobHandler
from ert_gui.ide.handlers.num_realizations_handler import NumRealizationsHandler
from ert_gui.ide.handlers.queue_option_handler import QueueOptionHandler
from ert_gui.ide.handlers.queue_system_handler import QueueSystemHandler
from ert_gui.ide.handlers.schedule_file_handler import ScheduleFileHandler
from ert_gui.ide.handlers.unknown_handler import UnknownHandler
from ert_gui.ide.keyword import Keyword


class KeywordHighlighter(QSyntaxHighlighter):
    COMMENT_STATE = 1
    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)

        self.comment_pattern = r'.*?(--.*)'
        self.keyword_pattern = re.compile('^\s*([A-Z_]+)\s')
        self.parameter_pattern = re.compile('\s+?(\S+)\s*?')

        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor(0, 128, 0))
        self.comment_format.setFontItalic(True)

        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor(200, 100, 0))
        # self.keyword_format.setFontWeight(QFont.Bold)

        self.error_format = QTextCharFormat()
        # self.error_format.setForeground(QColor(255, 0, 0))
        self.error_format.setUnderlineStyle(QTextCharFormat.WaveUnderline)
        self.error_format.setUnderlineColor(QColor(255, 0, 0))

        self.search_format = QTextCharFormat()
        self.search_format.setBackground(QColor(220, 220, 220))

        self.builtin_format = QTextCharFormat()
        self.builtin_format.setForeground(QColor(0, 170, 227))

        self.search_string = ""

        self.handlers = []
        self.handlers.append(DefineHandler())
        self.handlers.append(QueueOptionHandler())
        self.handlers.append(QueueSystemHandler())
        self.handlers.append(NumRealizationsHandler())
        self.handlers.append(InstallJobHandler())
        self.handlers.append(GridHandler())
        self.handlers.append(ScheduleFileHandler())
        self.handlers.append(InitSectionHandler())
        self.handlers.append(DataFileHandler())

        self.handler_names = [handler.keyword_name for handler in self.handlers]

        self.handlers.append(UnknownHandler())


    def highlightBlock(self, complete_block):
        block = unicode(complete_block)

        comment_match = re.match(self.comment_pattern, block)
        if comment_match is not None:
            self.setFormat(comment_match.start(1), comment_match.end(1), self.comment_format)
            block = block[0:comment_match.start(1)]


        keyword_match = re.match(self.keyword_pattern, block)

        if keyword_match is not None:
            value_match = self.parameter_pattern.finditer(block, keyword_match.end(1))

            keyword = Keyword(keyword_match.group(1), keyword_match.start(1), keyword_match.end(1))

            for match in value_match:
                keyword.addParameter(match.group(1), match.start(1), match.end(1))

            for handler in self.handlers:
                if handler.isHandlerFor(keyword):
                    handler.handle(keyword, self)
                    break


        elif comment_match is None and keyword_match is None and len(block.strip()) > 0:
            self.setFormat(0, len(block), self.error_format)

        if self.search_string != "":
            for match in re.finditer("(%s)" % self.search_string, complete_block):
                print(match.group(1), match.start(1), match.end(1))
                self.setFormat(match.start(1), match.end(1) - match.start(1), self.search_format)


    def setSearchString(self, string):
        if self.search_string != unicode(string):
            self.search_string = unicode(string)
            self.rehighlight()


