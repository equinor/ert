from ert.test import ErtTestContext
from ert.test.extended_testcase import ExtendedTestCase
from ert.util import BoolVector


class LoadResultsManuallyTest(ExtendedTestCase):
    def setUp(self):
        self.config_file = self.createTestPath("Statoil/config/with_data/config")

    def test_load_results_manually(self):
        with ErtTestContext("manual_load_test", self.config_file) as test_context:
            ert = test_context.getErt()
            load_into_case = "A1"
            load_from_case = "default"

            load_into =  ert.getEnkfFsManager().getFileSystem(load_into_case)
            load_from =  ert.getEnkfFsManager().getFileSystem(load_from_case)

            ert.getEnkfFsManager().switchFileSystem(load_from)
            realisations = BoolVector(default_value=True,initial_size=25)
            iteration = 0


            ert.loadFromForwardModel(realisations, iteration, load_into)

            load_from_case_state_map = load_from.getStateMap()
            load_into_case_state_map = load_into.getStateMap()

            load_from_states = [state for state in load_from_case_state_map]
            load_into_states = [state for state in load_into_case_state_map]


            self.assertListEqual(load_from_states, load_into_states)










