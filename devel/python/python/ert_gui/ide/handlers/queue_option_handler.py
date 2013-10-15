import re
from ert_gui.ide.handlers.keyword_handler import KeywordHandler


class QueueOptionHandler(KeywordHandler):
    QUEUE_TYPE_PATTERN = re.compile("^LSF|RSH|LOCAL|TORQUE$")

    def __init__(self):
        super(QueueOptionHandler, self).__init__("QUEUE_OPTION")
        self.setExpectedParameterCount(3)

    def handle(self, keyword, highlighter):
        """
        @type keyword: Keyword
        @type highlighter: QSyntaxHighlighter
        """

        if keyword.parameterCount() > 3:
            keyword.mergeParameters(3) # hack, should have separate parsers for each keyword type

        if keyword.parameterCount() > 0:
            queue = keyword[0]

            if self.QUEUE_TYPE_PATTERN.match(queue.value) is None:
                queue.error = True
                queue.error_message = "Parameter should be one of: LSF, RSH, LOCAL or TORQUE."
            else:
                highlighter.setFormat(queue.start, queue.length, highlighter.builtin_format)


        if keyword.parameterCount() == 3:
            key = keyword[1]
            value = keyword[2]


        super(QueueOptionHandler, self).handle(keyword, highlighter)


    def parameterOptions(self, keyword, prefix, position_in_block):
        index = keyword.parameterIndexForPosition(position_in_block)

        if index == 0:
            return ["LSF", "LOCAL", "RSH", "TORQUE"]

        return []