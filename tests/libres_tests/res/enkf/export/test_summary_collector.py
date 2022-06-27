from libres_utils import ResTest
from pytest import MonkeyPatch

from res.enkf.export import SummaryCollector
from res.test import ErtTestContext
import pytest


class SummaryCollectorTest(ResTest):
    def setUp(self):
        self.monkeypatch = MonkeyPatch()
        self.monkeypatch.setenv(
            "TZ", "CET"
        )  # The ert_statoil case was generated in CET
        self.config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def tearDown(self):
        self.monkeypatch.undo()

    def test_summary_collector(self):
        with ErtTestContext(self.config) as context:
            ert = context.getErt()

            data = SummaryCollector.loadAllSummaryData(ert, "default_0")

            self.assertFloatEqual(data["WWCT:OP2"][0]["2010-01-10"], 0.385549)
            self.assertFloatEqual(data["WWCT:OP2"][24]["2010-01-10"], 0.498331)

            self.assertFloatEqual(data["FOPR"][0]["2010-01-10"], 0.118963)
            self.assertFloatEqual(data["FOPR"][0]["2015-06-23"], 0.133601)

            # pylint: disable=pointless-statement
            # realization 20:
            data.loc[20]

            with self.assertRaises(KeyError):
                # realization 60:
                data.loc[60]

            data = SummaryCollector.loadAllSummaryData(
                ert, "default_0", ["WWCT:OP1", "WWCT:OP2"]
            )

            self.assertFloatEqual(data["WWCT:OP1"][0]["2010-01-10"], 0.352953)
            self.assertFloatEqual(data["WWCT:OP2"][0]["2010-01-10"], 0.385549)

            with self.assertRaises(KeyError):
                data["FOPR"]

            realization_index = 10
            data = SummaryCollector.loadAllSummaryData(
                ert,
                "default_0",
                ["WWCT:OP1", "WWCT:OP2"],
                realization_index=realization_index,
            )

            assert data.index.levels[0] == [realization_index]
            assert len(data.index.levels[1]) == 200
            assert list(data.columns) == ["WWCT:OP1", "WWCT:OP2"]

            non_existing_realization_index = 150
            with pytest.raises(IndexError):
                data = SummaryCollector.loadAllSummaryData(
                    ert,
                    "default_0",
                    ["WWCT:OP1", "WWCT:OP2"],
                    realization_index=non_existing_realization_index,
                )
