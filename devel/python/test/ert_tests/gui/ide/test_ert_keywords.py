from ert_gui.ide.keywords.definitions import IntegerArgument, StringArgument, BoolArgument
from ert_gui.ide.keywords import ErtKeywords
from ert_tests import ExtendedTestCase


class ErtKeywordTest(ExtendedTestCase):

    def setUp(self):
        self.keywords = ErtKeywords()

    def keywordTest(self, name, argument_types, documentation_link, group, required=False):
        self.assertTrue(name in self.keywords)

        cld = self.keywords[name]

        self.assertEqual(cld.keywordDefinition().name(), name)
        self.assertEqual(cld.group(), group)
        self.assertEqual(cld.documentationLink(), documentation_link)
        self.assertEqual(cld.isRequired(), required)

        arguments = cld.argumentDefinitions()

        self.assertEqual(len(arguments), len(argument_types))

        for index in range(len(arguments)):
            self.assertIsInstance(arguments[index], argument_types[index])



    def test_ensemble_keywords(self):
        self.keywordTest("NUM_REALIZATIONS", [IntegerArgument], "ensemble/num_realizations", "Ensemble", True)


    def test_queue_system_keywords(self):
        self.keywordTest("QUEUE_SYSTEM", [StringArgument], "queue_system/queue_system", "Queue System")
        self.keywordTest("QUEUE_OPTION", [StringArgument, StringArgument, StringArgument], "queue_system/queue_option", "Queue System")


    def test_control_simulations_keywords(self):
        self.keywordTest("MAX_RUNTIME", [IntegerArgument], "control_simulations/max_runtime", "Simulation Control")
        self.keywordTest("MIN_REALIZATIONS", [IntegerArgument], "control_simulations/min_realizations", "Simulation Control")
        self.keywordTest("STOP_LONG_RUNNING", [BoolArgument], "control_simulations/stop_long_running", "Simulation Control")


#


