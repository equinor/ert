from tests import ResTest
from res.test import ErtTestContext

from res.enkf.export import MisfitCollector


class MisfitCollectorTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def test_misfit_collector(self):
        with ErtTestContext("python/enkf/export/misfit_collector", self.config) as context:
            ert = context.getErt()
            data = MisfitCollector.loadAllMisfitData(ert, "default_0")

            self.assertFloatEqual(data["MISFIT:FOPR"][0],  737.436374)
            self.assertFloatEqual(data["MISFIT:FOPR"][24], 1258.644538)

            self.assertFloatEqual(data["MISFIT:TOTAL"][0], 765.709246)
            self.assertFloatEqual(data["MISFIT:TOTAL"][24], 1357.730551)

            realization_20 = data.loc[20]

            with self.assertRaises(KeyError):
                realization_60 = data.loc[60]
