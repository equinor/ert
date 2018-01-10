from res.util.enums import MessageLevelEnum

from tests import ResTest

class LogTest(ResTest):

    def test_enums(self):
        self.assertEnumIsFullyDefined(MessageLevelEnum, "message_level_type", "libres_util/include/ert/res_util/log.h")
