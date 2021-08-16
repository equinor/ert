from ert_shared.ide.keywords.definitions import (
    StringArgument,
    KeywordDefinition,
    IntegerArgument,
    PathArgument,
)
from ert_shared.ide.keywords.definitions.configuration_line_definition import (
    ConfigurationLineDefinition,
)


class QueueSystemKeywords(object):
    def __init__(self, ert_keywords):
        super(QueueSystemKeywords, self).__init__()
        self.group = "Queue System"

        ert_keywords.addKeyword(self.addQueueOption())
        ert_keywords.addKeyword(self.addQueueSystem())
        ert_keywords.addKeyword(self.addLsfServer())
        ert_keywords.addKeyword(self.addLsfQueue())
        ert_keywords.addKeyword(self.addTorqueQueue())
        ert_keywords.addKeyword(self.addRshHost())
        ert_keywords.addKeyword(self.addRshCommand())
        ert_keywords.addKeyword(self.addHostType())
        ert_keywords.addKeyword(self.addLsfResource())

    def addQueueOption(self):
        queue_option = ConfigurationLineDefinition(
            keyword=KeywordDefinition("QUEUE_OPTION"),
            arguments=[
                StringArgument(built_in=True),
                StringArgument(built_in=True),
                StringArgument(allow_space=True, rest_of_line=True),
            ],
            documentation_link="keywords/queue_option",
            group=self.group,
        )

        return queue_option

    def addQueueSystem(self):
        queue_system = ConfigurationLineDefinition(
            keyword=KeywordDefinition("QUEUE_SYSTEM"),
            arguments=[StringArgument(built_in=True)],
            documentation_link="keywords/queue_system",
            group=self.group,
        )
        return queue_system

    def addLsfServer(self):
        lsf_server = ConfigurationLineDefinition(
            keyword=KeywordDefinition("LSF_SERVER"),
            arguments=[StringArgument(built_in=True)],
            documentation_link="keywords/lsf_server",
            group=self.group,
        )
        return lsf_server

    def addLsfQueue(self):
        lsf_queue = ConfigurationLineDefinition(
            keyword=KeywordDefinition("LSF_QUEUE"),
            arguments=[StringArgument()],
            documentation_link="keywords/lsf_queue",
            group=self.group,
        )
        return lsf_queue

    def addLsfResource(self):
        lsf_resource = ConfigurationLineDefinition(
            keyword=KeywordDefinition("LSF_RESOURCE"),
            arguments=[StringArgument(rest_of_line=True, allow_space=True)],
            documentation_link="keywords/lsf_resource",
            group=self.group,
        )
        return lsf_resource

    def addTorqueQueue(self):
        torque_queue = ConfigurationLineDefinition(
            keyword=KeywordDefinition("TORQUE_QUEUE"),
            arguments=[StringArgument()],
            documentation_link="keywords/torque_queue",
            group=self.group,
        )
        return torque_queue

    def addRshHost(self):
        rsh_host = ConfigurationLineDefinition(
            keyword=KeywordDefinition("RSH_HOST"),
            arguments=[
                StringArgument(),
                StringArgument(rest_of_line=True, allow_space=True),
            ],
            documentation_link="keywords/rsh_host",
            group=self.group,
        )
        return rsh_host

    def addRshCommand(self):
        rsh_command = ConfigurationLineDefinition(
            keyword=KeywordDefinition("RSH_COMMAND"),
            arguments=[PathArgument()],
            documentation_link="keywords/rsh_command",
            group=self.group,
        )
        return rsh_command

    def addHostType(self):
        host_type = ConfigurationLineDefinition(
            keyword=KeywordDefinition("HOST_TYPE"),
            arguments=[StringArgument(rest_of_line=True, allow_space=True)],
            documentation_link="keywords/host_type",
            group=self.group,
        )
        return host_type
