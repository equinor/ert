import os
import shutil

from pandas import DataFrame

from ert_gui.tools.plot.plot_api import PlotApi
from res.enkf import EnKFMain, ResConfig

from ert_shared.libres_facade import LibresFacade
from tests.utils import SOURCE_DIR, tmpdir
from unittest import TestCase


class PlotApiTest(TestCase):

    def api(self):
        config_file = 'snake_oil.ert'

        rc = ResConfig(user_config_file=config_file)
        rc.convertToCReference(None)
        ert = EnKFMain(rc)
        facade = LibresFacade(ert)
        api = PlotApi(facade)
        return api

    @tmpdir(os.path.join(SOURCE_DIR, 'test-data/local/snake_oil'))
    def test_all_keys_present(self):
        api = self.api()

        key_defs = api.all_data_type_keys()
        keys = {x["key"] for x in key_defs}
        expected = {'BPR:1,3,8', 'BPR:445', 'BPR:5,5,5', 'BPR:721', 'FGIP', 'FGIPH', 'FGOR', 'FGORH', 'FGPR', 'FGPRH',
                    'FGPT', 'FGPTH', 'FOIP', 'FOIPH', 'FOPR', 'FOPRH', 'FOPT', 'FOPTH', 'FWCT', 'FWCTH', 'FWIP',
                    'FWIPH', 'FWPR', 'FWPRH', 'FWPT', 'FWPTH', 'TIME', 'WGOR:OP1', 'WGOR:OP2', 'WGORH:OP1', 'WGORH:OP2',
                    'WGPR:OP1', 'WGPR:OP2', 'WGPRH:OP1', 'WGPRH:OP2', 'WOPR:OP1', 'WOPR:OP2', 'WOPRH:OP1', 'WOPRH:OP2',
                    'WWCT:OP1', 'WWCT:OP2', 'WWCTH:OP1', 'WWCTH:OP2', 'WWPR:OP1', 'WWPR:OP2', 'WWPRH:OP1', 'WWPRH:OP2',
                    'SNAKE_OIL_PARAM:BPR_138_PERSISTENCE', 'SNAKE_OIL_PARAM:BPR_555_PERSISTENCE',
                    'SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE', 'SNAKE_OIL_PARAM:OP1_OCTAVES', 'SNAKE_OIL_PARAM:OP1_OFFSET',
                    'SNAKE_OIL_PARAM:OP1_PERSISTENCE', 'SNAKE_OIL_PARAM:OP2_DIVERGENCE_SCALE',
                    'SNAKE_OIL_PARAM:OP2_OCTAVES', 'SNAKE_OIL_PARAM:OP2_OFFSET', 'SNAKE_OIL_PARAM:OP2_PERSISTENCE',
                    'SNAKE_OIL_GPR_DIFF@199', 'SNAKE_OIL_OPR_DIFF@199', 'SNAKE_OIL_WPR_DIFF@199'}
        self.assertSetEqual(expected, keys)

    @tmpdir(os.path.join(SOURCE_DIR, 'test-data/local/snake_oil'))
    def test_observation_key_present(self):
        api = self.api()
        key_defs = api.all_data_type_keys()
        expected_obs = {
            'FOPR': ['FOPR'],
            'WOPR:OP1': ['WOPR_OP1_108', 'WOPR_OP1_190', 'WOPR_OP1_144', 'WOPR_OP1_9', 'WOPR_OP1_72', 'WOPR_OP1_36'],
            'SNAKE_OIL_WPR_DIFF@199': ["WPR_DIFF_1"]
        }

        for key_def in key_defs:
            if key_def["key"] in expected_obs:
                expected = expected_obs[key_def["key"]]
                self.assertEqual(expected, key_def["observations"])
            else:
                self.assertEqual(0, len(key_def["observations"]))

    @tmpdir(os.path.join(SOURCE_DIR, 'test-data/local/snake_oil'))
    def test_can_load_data_and_observations(self):
        api = self.api()
        key_defs = api.all_data_type_keys()
        cases = api.get_all_cases_not_running()
        for case in cases:
            for key_def in key_defs:
                obs = key_def["observations"]
                obs_data = api.observations_for_obs_keys(case["name"], obs)
                data = api.data_for_key(case["name"], key_def["key"])

                self.assertIsInstance(data, DataFrame)
                self.assertTrue(not data.empty)

                self.assertIsInstance(obs_data, DataFrame)
                if len(obs) > 0:
                    self.assertTrue(not obs_data.empty)

    @tmpdir(os.path.join(SOURCE_DIR, 'test-data/local/snake_oil'))
    def test_no_storage(self):
        shutil.rmtree("storage")
        api = self.api()
        key_defs = api.all_data_type_keys()
        cases = api.get_all_cases_not_running()
        for case in cases:
            for key_def in key_defs:
                obs = key_def["observations"]
                obs_data = api.observations_for_obs_keys(case["name"], obs)
                data = api.data_for_key(case["name"], key_def["key"])
                self.assertIsInstance(obs_data, DataFrame)
                self.assertIsInstance(data, DataFrame)
                self.assertTrue(data.empty)

    @tmpdir(os.path.join(SOURCE_DIR, 'test-data/local/snake_oil'))
    def test_key_def_structure(self):
        shutil.rmtree("storage")
        api = self.api()
        key_defs = api.all_data_type_keys()
        fopr = next(x for x in key_defs if x["key"] == "FOPR")

        expected = {
            'dimensionality': 2,
            'has_refcase': True,
            'index_type': 'VALUE',
            'key': 'FOPR',
            'metadata': {'data_origin': 'Summary'},
            'observations': ['FOPR'],
            'log_scale': False,
        }

        self.assertEqual(expected, fopr)

    @tmpdir(os.path.join(SOURCE_DIR, 'test-data/local/snake_oil'))
    def test_case_structure(self):
        api = self.api()
        cases = api.get_all_cases_not_running()
        case = next(x for x in cases if x["name"] == "default_0")

        expected = {
            'has_data': True,
            'hidden': False,
            'name': 'default_0'
        }

        self.assertEqual(expected, case)
