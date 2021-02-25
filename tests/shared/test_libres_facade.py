import os
from pandas.core.base import PandasObject

from res.enkf import EnKFMain, ResConfig

from ert_shared.libres_facade import LibresFacade
from tests.utils import SOURCE_DIR, tmpdir
from unittest import TestCase


class LibresFacadeTest(TestCase):
    def facade(self):
        config_file = "snake_oil.ert"

        rc = ResConfig(user_config_file=config_file)
        rc.convertToCReference(None)
        ert = EnKFMain(rc)
        facade = LibresFacade(ert)
        return facade

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_keyword_type_checks(self):
        facade = self.facade()
        self.assertTrue(facade.is_gen_data_key("SNAKE_OIL_GPR_DIFF@199"))
        self.assertTrue(facade.is_summary_key("BPR:1,3,8"))
        self.assertTrue(facade.is_gen_kw_key("SNAKE_OIL_PARAM:BPR_138_PERSISTENCE"))

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_keyword_type_checks_missing_key(self):
        facade = self.facade()
        self.assertFalse(facade.is_gen_data_key("nokey"))
        self.assertFalse(facade.is_summary_key("nokey"))
        self.assertFalse(facade.is_gen_kw_key("nokey"))

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_data_fetching(self):
        facade = self.facade()
        data = [
            facade.gather_gen_data_data("default_0", "SNAKE_OIL_GPR_DIFF@199"),
            facade.gather_summary_data("default_0", "BPR:1,3,8"),
            facade.gather_gen_kw_data(
                "default_0", "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE"
            ),
        ]

        for dataframe in data:
            self.assertIsInstance(dataframe, PandasObject)
            self.assertFalse(dataframe.empty)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_data_fetching_missing_case(self):
        facade = self.facade()
        data = [
            facade.gather_gen_data_data("nocase", "SNAKE_OIL_GPR_DIFF@199"),
            facade.gather_summary_data("nocase", "BPR:1,3,8"),
            facade.gather_gen_kw_data("nocase", "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE"),
        ]

        for dataframe in data:
            self.assertIsInstance(dataframe, PandasObject)
            self.assertTrue(dataframe.empty)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_data_fetching_missing_key(self):
        facade = self.facade()
        data = [
            facade.gather_gen_data_data("default_0", "nokey"),
            facade.gather_summary_data("default_0", "nokey"),
            facade.gather_gen_kw_data("default_0", "nokey"),
        ]

        for dataframe in data:
            self.assertIsInstance(dataframe, PandasObject)
            self.assertTrue(dataframe.empty)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_cases_list(self):
        facade = self.facade()
        cases = facade.cases()
        self.assertEqual(["default_0", "default_1"], cases)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_is_hidden(self):
        facade = self.facade()
        self.assertFalse(facade.is_case_hidden("default_0"))
        self.assertFalse(facade.is_case_hidden("nocase"))

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_has_data(self):
        facade = self.facade()
        self.assertTrue(facade.case_has_data("default_0"))
        self.assertFalse(facade.case_has_data("default"))

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_is_running(self):
        facade = self.facade()
        self.assertFalse(facade.is_case_running("default_0"))
        self.assertFalse(facade.is_case_running("nocase"))

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_all_data_type_keys(self):
        facade = self.facade()
        keys = facade.all_data_type_keys()

        expected = [
            "BPR:1,3,8",
            "BPR:445",
            "BPR:5,5,5",
            "BPR:721",
            "FGIP",
            "FGIPH",
            "FGOR",
            "FGORH",
            "FGPR",
            "FGPRH",
            "FGPT",
            "FGPTH",
            "FOIP",
            "FOIPH",
            "FOPR",
            "FOPRH",
            "FOPT",
            "FOPTH",
            "FWCT",
            "FWCTH",
            "FWIP",
            "FWIPH",
            "FWPR",
            "FWPRH",
            "FWPT",
            "FWPTH",
            "TIME",
            "WGOR:OP1",
            "WGOR:OP2",
            "WGORH:OP1",
            "WGORH:OP2",
            "WGPR:OP1",
            "WGPR:OP2",
            "WGPRH:OP1",
            "WGPRH:OP2",
            "WOPR:OP1",
            "WOPR:OP2",
            "WOPRH:OP1",
            "WOPRH:OP2",
            "WWCT:OP1",
            "WWCT:OP2",
            "WWCTH:OP1",
            "WWCTH:OP2",
            "WWPR:OP1",
            "WWPR:OP2",
            "WWPRH:OP1",
            "WWPRH:OP2",
            "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
            "SNAKE_OIL_PARAM:BPR_555_PERSISTENCE",
            "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
            "SNAKE_OIL_PARAM:OP1_OCTAVES",
            "SNAKE_OIL_PARAM:OP1_OFFSET",
            "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
            "SNAKE_OIL_PARAM:OP2_DIVERGENCE_SCALE",
            "SNAKE_OIL_PARAM:OP2_OCTAVES",
            "SNAKE_OIL_PARAM:OP2_OFFSET",
            "SNAKE_OIL_PARAM:OP2_PERSISTENCE",
            "SNAKE_OIL_GPR_DIFF@199",
            "SNAKE_OIL_OPR_DIFF@199",
            "SNAKE_OIL_WPR_DIFF@199",
        ]

        self.assertEqual(expected, keys)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_observation_keys(self):
        facade = self.facade()

        expected_obs = {
            "FOPR": ["FOPR"],
            "WOPR:OP1": [
                "WOPR_OP1_108",
                "WOPR_OP1_190",
                "WOPR_OP1_144",
                "WOPR_OP1_9",
                "WOPR_OP1_72",
                "WOPR_OP1_36",
            ],
            "SNAKE_OIL_WPR_DIFF@199": ["WPR_DIFF_1"],
        }

        for key in facade.all_data_type_keys():
            obs_keys = facade.observation_keys(key)
            expected = []
            if key in expected_obs:
                expected = expected_obs[key]
            self.assertEqual(expected, obs_keys)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_observation_keys_missing_key(self):
        facade = self.facade()
        obs_keys = facade.observation_keys("nokey")
        self.assertEqual([], obs_keys)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_has_refcase(self):
        facade = self.facade()
        self.assertTrue(facade.has_refcase("FOPR"))

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_has_refcase_missing_key(self):
        facade = self.facade()
        self.assertFalse(facade.has_refcase("nokey"))

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_refcase_data(self):
        facade = self.facade()
        data = facade.refcase_data("FOPR")
        self.assertIsInstance(data, PandasObject)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_refcase_data_missing_key(self):
        facade = self.facade()
        data = facade.refcase_data("nokey")
        self.assertIsInstance(data, PandasObject)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_history_data(self):
        facade = self.facade()
        data = facade.history_data("FOPR")
        self.assertIsInstance(data, PandasObject)

        facade = self.facade()
        data = facade.history_data("WOPR:OP1")
        self.assertIsInstance(data, PandasObject)

    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/snake_oil"))
    def test_case_history_data_missing_key(self):
        facade = self.facade()
        data = facade.history_data("nokey")
        self.assertIsInstance(data, PandasObject)
