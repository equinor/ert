from ert_gui.ide.keywords.definitions import IntegerArgument, StringArgument
from ert_gui.ide.keywords import ErtKeywords
from ert_tests import ExtendedTestCase


class ErtKeywordTest(ExtendedTestCase):
#
#
    def test_num_realizations(self):
        keywords = ErtKeywords()

        self.assertTrue("NUM_REALIZATIONS" in keywords)

        #: :type: ConfigurationLineDefinition
        num_realizations = keywords["NUM_REALIZATIONS"]


        self.assertEqual(num_realizations.documentLink(), "ensemble/num_realizations")
        self.assertEqual(num_realizations.group(), "Ensemble")
        self.assertEqual(num_realizations.keywordDefinition().name(), "NUM_REALIZATIONS")
        self.assertIsInstance(num_realizations.argumentDefinitions()[0], IntegerArgument)
        self.assertTrue(num_realizations.isRequired())


    def test_queue_option(self):
        keywords = ErtKeywords()

        self.assertTrue("QUEUE_SYSTEM" in keywords)
        keyword = keywords["QUEUE_SYSTEM"]

        self.assertEqual(keyword.documentLink(), "queue_system/queue_system")
        self.assertEqual(keyword.group(), "Queue System")

#     def test_num_realizations(self):
#
#         test_line = "NUM_REALIZATIONS 45"
#
#
#     def test_queue_option(self):
#         keywords = ErtKeywords()
#
#         self.assertTrue("QUEUE_OPTION" in keywords)
#
#         queue_option = keywords["QUEUE_OPTION"]
#
#         self.assertEqual(queue_option.name, "QUEUE_OPTION")
#         self.assertEqual(queue_option.documentation, "queue_system/queue_option")
#         self.assertEqual(queue_option.group, "Queue System")
#         self.assertIsInstance(queue_option.arguments[0], StringArgument)
#         self.assertIsInstance(queue_option.arguments[1], StringArgument)
#         self.assertIsInstance(queue_option.arguments[2], StringArgument)
#
#         self.assertTrue(queue_option.arguments[0].isBuiltIn())
#         self.assertTrue(queue_option.arguments[1].isBuiltIn())
#         self.assertFalse(queue_option.arguments[2].isBuiltIn())
#
#
#


