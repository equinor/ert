import os
import logging
import pytest
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from pandas.core.base import PandasObject

from utils import SOURCE_DIR
from ert_utils import tmpdir

from ert_shared.libres_facade import LibresFacade
from res.enkf import EnKFMain, ResConfig

# This hack allows us to call the original method when
# mocking loadAllSummaryData()
from res.enkf.export import SummaryCollector

_orig_loader = SummaryCollector.loadAllSummaryData


# Define this in the module-scope. If defined as a class-method
# it passes "self" when calling _orig_loader. There are probably
# ways to handle this, but it's simpler to define it here
def _add_duplicate_row(*args, **kwargs):
    df = _orig_loader(*args, **kwargs)
    # Append copy of last date to each realization
    idx = pd.MultiIndex.from_tuples(
        [(i, df.loc[i].iloc[-1].name) for i in df.index.levels[0]]
    )
    df_new = pd.DataFrame(0, index=idx, columns=df.columns)
    return pd.concat([df, df_new]).sort_index()


class LibresFacadeTest(TestCase):
    def facade(self):
        config_file = "snake_oil.ert"

        rc = ResConfig(user_config_file=config_file)
        rc.convertToCReference(None)
        ert = EnKFMain(rc)
        facade = LibresFacade(ert)
        return facade

    # Since "caplog "is a pytest-fixture and this class is based on Python
    # unittest, use this trick from https://stackoverflow.com/a/50375022
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_keyword_type_checks(self):
        facade = self.facade()
        self.assertTrue(facade.is_gen_data_key("SNAKE_OIL_GPR_DIFF@199"))
        self.assertTrue(facade.is_summary_key("BPR:1,3,8"))
        self.assertTrue(facade.is_gen_kw_key("SNAKE_OIL_PARAM:BPR_138_PERSISTENCE"))

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_keyword_type_checks_missing_key(self):
        facade = self.facade()
        self.assertFalse(facade.is_gen_data_key("nokey"))
        self.assertFalse(facade.is_summary_key("nokey"))
        self.assertFalse(facade.is_gen_kw_key("nokey"))

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
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

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
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

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
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

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_cases_list(self):
        facade = self.facade()
        cases = facade.cases()
        self.assertEqual(["default_0", "default_1"], cases)

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_is_hidden(self):
        facade = self.facade()
        self.assertFalse(facade.is_case_hidden("default_0"))
        self.assertFalse(facade.is_case_hidden("nocase"))

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_has_data(self):
        facade = self.facade()
        self.assertTrue(facade.case_has_data("default_0"))
        self.assertFalse(facade.case_has_data("default"))

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_is_running(self):
        facade = self.facade()
        self.assertFalse(facade.is_case_running("default_0"))
        self.assertFalse(facade.is_case_running("nocase"))

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
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

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
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

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_observation_keys_missing_key(self):
        facade = self.facade()
        obs_keys = facade.observation_keys("nokey")
        self.assertEqual([], obs_keys)

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_has_refcase(self):
        facade = self.facade()
        self.assertTrue(facade.has_refcase("FOPR"))

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_has_refcase_missing_key(self):
        facade = self.facade()
        self.assertFalse(facade.has_refcase("nokey"))

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_refcase_data(self):
        facade = self.facade()
        data = facade.refcase_data("FOPR")
        self.assertIsInstance(data, PandasObject)

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_refcase_data_missing_key(self):
        facade = self.facade()
        data = facade.refcase_data("nokey")
        self.assertIsInstance(data, PandasObject)

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_history_data(self):
        facade = self.facade()
        data = facade.history_data("FOPR")
        self.assertIsInstance(data, PandasObject)

        facade = self.facade()
        data = facade.history_data("WOPR:OP1")
        self.assertIsInstance(data, PandasObject)

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_case_history_data_missing_key(self):
        facade = self.facade()
        data = facade.history_data("nokey")
        self.assertIsInstance(data, PandasObject)

    def _do_verify_indices_and_values(self, data):
        # Verify indices
        assert data.columns.name == "Realization"
        assert all(data.columns == range(25))
        assert data.index.name == "Date"
        assert all(data.index == pd.date_range("2010-01-10", periods=200, freq="10D"))

        # Verify selected datapoints
        assert data.iloc[0][0] == pytest.approx(0.118963, abs=1e-6)  # top-left
        assert data.iloc[199][0] == pytest.approx(0.133601, abs=1e-6)  # bottom-left
        assert data.iloc[4][9] == pytest.approx(
            0.178028, abs=1e-6
        )  # somewhere in the middle
        # bottom-right 5 entries in col
        assert data.iloc[-5:][24].values == pytest.approx(
            [0.143714, 0.142230, 0.140191, 0.140143, 0.139711], abs=1e-6
        )

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    def test_summary_data_verify_indices_and_values(self):
        facade = self.facade()
        self._caplog.clear()
        with self._caplog.at_level(logging.WARNING):
            self._do_verify_indices_and_values(
                facade.gather_summary_data("default_0", "FOPR")
            )
            assert "contains duplicate timestamps" not in self._caplog.text

    @tmpdir(SOURCE_DIR / "test-data/local/snake_oil")
    @patch(
        "res.enkf.export.SummaryCollector.loadAllSummaryData", wraps=_add_duplicate_row
    )
    def test_summary_data_verify_remove_duplicates(self, *args):
        facade = self.facade()
        self._caplog.clear()
        with self._caplog.at_level(logging.WARNING):
            self._do_verify_indices_and_values(
                facade.gather_summary_data("default_0", "FOPR")
            )
            assert "contains duplicate timestamps" in self._caplog.text
