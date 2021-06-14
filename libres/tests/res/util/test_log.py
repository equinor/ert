from utils import ResTest

from res.util.enums import MessageLevelEnum


class LogTest(ResTest):
    def test_enums(self):
        self.assertEnumIsFullyDefined(
            MessageLevelEnum, "message_level_type", "lib/include/ert/res_util/log.hpp"
        )
