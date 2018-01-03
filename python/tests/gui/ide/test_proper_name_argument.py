from ert_gui.ide.keywords.definitions import ProperNameArgument
from tests import ResTest


class ProperNameArgumentTest(ResTest):

    def test_proper_name_argument(self):

        argument = ProperNameArgument()

        self.assertTrue(argument.validate("NAME"))
        self.assertTrue(argument.validate("__NAME__"))
        self.assertTrue(argument.validate("<NAME>"))
        self.assertTrue(argument.validate("-NAME-"))

        self.assertFalse(argument.validate("-NA ME-"))
        self.assertFalse(argument.validate("NAME*"))

