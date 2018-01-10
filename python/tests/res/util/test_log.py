from ecl.util.enums import MessageLevelEnum

from tests import EclTest

class LogTest(EclTest):

    def test_enums(self):
        self.assertEnumIsFullyDefined(MessageLevelEnum, "message_level_type", "lib/include/ert/util/log.h")
