import pytest

from ....libres_utils import ResTest

from ert._c_wrappers.enkf.export import MisfitCollector
from ert._c_wrappers.test import ErtTestContext


class MisfitCollectorTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def test_misfit_collector(self):
        with ErtTestContext(self.config) as context:
            ert = context.getErt()
            data = MisfitCollector.loadAllMisfitData(ert, "default_0")

            assert pytest.approx(data["MISFIT:FOPR"][0]) == 738.735586
            assert pytest.approx(data["MISFIT:FOPR"][24]) == 1260.086789

            assert pytest.approx(data["MISFIT:TOTAL"][0]) == 767.008457
            assert pytest.approx(data["MISFIT:TOTAL"][24]) == 1359.172803

            # pylint: disable=pointless-statement
            # realization 20:
            data.loc[20]

            with self.assertRaises(KeyError):
                # realization 60:
                data.loc[60]
