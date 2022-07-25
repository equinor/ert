import pytest

from ....libres_utils import ResTest

from res.enkf.export import SummaryObservationCollector
from res.test import ErtTestContext


class SummaryObservationCollectorTest(ResTest):
    def setUp(self):
        self.monkeypatch = pytest.MonkeyPatch()
        self.monkeypatch.setenv(
            "TZ", "CET"
        )  # The ert_statoil case was generated in CET
        self.config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def tearDown(self):
        self.monkeypatch.undo()

    def test_summary_observation_collector(self):

        with ErtTestContext(self.config) as context:

            ert = context.getErt()

            self.assertTrue(
                SummaryObservationCollector.summaryKeyHasObservations(ert, "FOPR")
            )
            self.assertFalse(
                SummaryObservationCollector.summaryKeyHasObservations(ert, "FOPT")
            )

            keys = SummaryObservationCollector.getAllObservationKeys(ert)
            self.assertTrue("FOPR" in keys)
            self.assertTrue("WOPR:OP1" in keys)
            self.assertFalse("WOPR:OP2" in keys)

            data = SummaryObservationCollector.loadObservationData(ert, "default_0")

            assert pytest.approx(data["FOPR"]["2010-01-10"]) == 0.001696887
            assert pytest.approx(data["STD_FOPR"]["2010-01-10"]) == 0.1

            assert pytest.approx(data["WOPR:OP1"]["2010-03-31"]) == 0.1
            assert pytest.approx(data["STD_WOPR:OP1"]["2010-03-31"]) == 0.05

            # pylint: disable=pointless-statement
            with self.assertRaises(KeyError):
                data["FGIR"]

            data = SummaryObservationCollector.loadObservationData(
                ert, "default_0", ["WOPR:OP1"]
            )

            assert pytest.approx(data["WOPR:OP1"]["2010-03-31"]) == 0.1
            assert pytest.approx(data["STD_WOPR:OP1"]["2010-03-31"]) == 0.05

            with self.assertRaises(KeyError):
                data["FOPR"]
