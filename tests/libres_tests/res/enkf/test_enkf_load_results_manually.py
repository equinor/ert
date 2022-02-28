import os
import logging
import pytest
from ecl.util.util import BoolVector
from libres_utils import ResTest, tmpdir

from res.enkf.enums.realization_state_enum import RealizationStateEnum
from res.test import ErtTestContext
from res.enkf import EnKFMain


@pytest.mark.unstable
@pytest.mark.equinor_test
class LoadResultsManuallyTest(ResTest):
    def setUp(self):
        self.config_file = self.createTestPath("Equinor/config/with_data/config")

    @tmpdir()
    def test_load_results_manually(self):
        with ErtTestContext("manual_load_test", self.config_file) as test_context:
            ert = test_context.getErt()
            load_into_case = "A1"
            load_from_case = "default"

            load_into = ert.getEnkfFsManager().getFileSystem(load_into_case)
            load_from = ert.getEnkfFsManager().getFileSystem(load_from_case)

            ert.getEnkfFsManager().switchFileSystem(load_from)
            realisations = BoolVector(default_value=True, initial_size=25)
            realisations[7] = False
            iteration = 0

            loaded = ert.loadFromForwardModel(realisations, iteration, load_into)

            load_into_case_state_map = load_into.getStateMap()

            load_into_states = [state for state in load_into_case_state_map]

            expected = [RealizationStateEnum.STATE_HAS_DATA] * 25
            expected[7] = RealizationStateEnum.STATE_UNDEFINED

            self.assertListEqual(load_into_states, expected)
            self.assertEqual(24, loaded)
            self.assertEqual(25, len(expected))
            self.assertEqual(25, len(realisations))

    @tmpdir()
    def test_load_results_from_run_context(self):
        with ErtTestContext("manual_load_test", self.config_file) as test_context:
            ert = test_context.getErt()
            load_into_case = "A1"
            load_from_case = "default"

            load_into = ert.getEnkfFsManager().getFileSystem(load_into_case)
            load_from = ert.getEnkfFsManager().getFileSystem(load_from_case)

            ert.getEnkfFsManager().switchFileSystem(load_from)
            realisations = BoolVector(default_value=True, initial_size=25)
            realisations[7] = False
            iteration = 0

            run_context = ert.getRunContextENSEMPLE_EXPERIMENT(load_into, realisations)

            loaded = ert.loadFromRunContext(run_context, load_into)

            load_into_case_state_map = load_into.getStateMap()
            load_into_states = [state for state in load_into_case_state_map]

            expected = [RealizationStateEnum.STATE_HAS_DATA] * 25
            expected[7] = RealizationStateEnum.STATE_UNDEFINED

            self.assertListEqual(load_into_states, expected)
            self.assertEqual(24, loaded)
            self.assertEqual(25, len(expected))
            self.assertEqual(25, len(realisations))


@pytest.mark.parametrize("lazy_load", [True, False])
def test_load_results_manually(setup_case, caplog, monkeypatch, lazy_load):
    """
    This little test does not depend on Equinor-data and only verifies
    the lazy_load flag in forward_load_context plus memory-logging
    """
    if lazy_load:
        monkeypatch.setenv("ERT_LAZY_LOAD_SUMMARYDATA", str(lazy_load))
    res_config = setup_case("local/snake_oil", "snake_oil.ert")
    ert = EnKFMain(res_config)
    load_from = ert.getEnkfFsManager().getFileSystem("default_0")
    ert.getEnkfFsManager().switchFileSystem(load_from)
    realisations = BoolVector(default_value=False, initial_size=25)
    realisations[0] = True  # only need one to test what we want
    with caplog.at_level(logging.INFO):
        loaded = ert.loadFromForwardModel(realisations, 0, load_from)
        assert 0 == loaded  # they will  in fact all fail, but that's ok
        assert f"lazy={lazy_load}".lower() in caplog.text
