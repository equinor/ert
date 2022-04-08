from libres_utils import ResTest

from res.enkf.export import MisfitCollector
from res.test import ErtTestContext


class MisfitCollectorTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def test_misfit_collector(self):
        with ErtTestContext(
            "python/enkf/export/misfit_collector", self.config
        ) as context:
            ert = context.getErt()
            data = MisfitCollector.loadAllMisfitData(ert, "default_0")

            self.assertFloatEqual(data["MISFIT:FOPR"][0], 738.735586)
            self.assertFloatEqual(data["MISFIT:FOPR"][24], 1260.086789)

            self.assertFloatEqual(data["MISFIT:TOTAL"][0], 767.008457)
            self.assertFloatEqual(data["MISFIT:TOTAL"][24], 1359.172803)

            # realization 20:
            data.loc[20]

            with self.assertRaises(KeyError):
                # realization 60:
                data.loc[60]
