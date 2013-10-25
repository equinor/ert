import re
from ert_gui.ide.handlers.keyword_handler import KeywordHandler


class QueueSystemHandler(KeywordHandler):
    QUEUE_TYPE_PATTERN = re.compile("^LSF|RSH|LOCAL|TORQUE$")

    def __init__(self):
        super(QueueSystemHandler, self).__init__("QUEUE_SYSTEM")
        self.setExpectedParameterCount(1)

    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() > 0:
            queue = keyword[0]

            if self.QUEUE_TYPE_PATTERN.match(queue.value) is None:
                queue.error = True
                queue.error_message = "Parameter should be one of: LSF, RSH, LOCAL or TORQUE."
            else:
                highlighter.setFormat(queue.start, queue.length, highlighter.builtin_format)


        super(QueueSystemHandler, self).handle(keyword, highlighter)

    def parameterOptions(self, keyword, prefix, position_in_block):
        index = keyword.parameterIndexForPosition(position_in_block)

        if index == 0:
            return ["LSF", "RSH", "LOCAL", "TORQUE"]

        return []


