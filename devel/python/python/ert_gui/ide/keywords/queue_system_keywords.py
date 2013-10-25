from ert_gui.ide.keywords.definitions import StringArgument, KeywordDefinition
from ert_gui.ide.keywords.definitions.configuration_line_definition import ConfigurationLineDefinition


class QueueSystemKeywords(object):


    def __init__(self, ert_keywords):
        super(QueueSystemKeywords, self).__init__()
        self.group = "Queue System"

        ert_keywords.addKeyword(self.addQueueOption())
        ert_keywords.addKeyword(self.addQueueSystem())

    def addQueueOption(self):
        queue_option = ConfigurationLineDefinition(keyword = KeywordDefinition("QUEUE_OPTION"),
                                                   arguments=[
                                                       StringArgument(built_in=True),
                                                       StringArgument(built_in=True),
                                                       StringArgument(allow_space=True, rest_of_line=True)
                                                   ],
                                                   documentation_link="queue_system/queue_option",
                                                   group=self.group)

        return queue_option



    def addQueueSystem(self):
        queue_system = ConfigurationLineDefinition(keyword = KeywordDefinition("QUEUE_SYSTEM"),
                                                   arguments=[StringArgument(built_in=True)],
                                                   documentation_link="queue_system/queue_system",
                                                   group=self.group)
        return queue_system



