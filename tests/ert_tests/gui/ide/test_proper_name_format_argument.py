from ert_shared.ide.keywords.definitions import ProperNameFormatArgument

from ...ert_utils import ErtTest


class ProperNameFormatArgumentTest(ErtTest):
    def test_proper_name_format_argument(self):

        argument = ProperNameFormatArgument()

        self.assertTrue(argument.validate("NAME%d"))
        self.assertTrue(argument.validate("__NA%dME__"))
        self.assertTrue(argument.validate("<NAME>%d"))
        self.assertTrue(argument.validate("%d-NAME-"))

        self.assertFalse(argument.validate("-%dNA ME-"))
        self.assertFalse(argument.validate("NAME*%d"))
