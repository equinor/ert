import pytest

from libres_utils import ResTest

from res.enkf.export import GenDataCollector
from res.test import ErtTestContext


class GenDataCollectorTest(ResTest):
    def test_gen_data_collector(self):
        config = self.createTestPath("local/snake_oil/snake_oil.ert")
        with ErtTestContext(config) as context:
            ert = context.getErt()

            with self.assertRaises(KeyError):
                GenDataCollector.loadGenData(ert, "default_0", "RFT_XX", 199)

            with self.assertRaises(ValueError):
                GenDataCollector.loadGenData(
                    ert, "default_0", "SNAKE_OIL_OPR_DIFF", 198
                )

            data1 = GenDataCollector.loadGenData(
                ert, "default_0", "SNAKE_OIL_OPR_DIFF", 199
            )

            assert pytest.approx(data1[0][0]) == -0.008206
            assert pytest.approx(data1[24][1]) == -0.119255
            assert pytest.approx(data1[24][1000]) == -0.258516

            realization_index = 10
            data1 = GenDataCollector.loadGenData(
                ert,
                "default_0",
                "SNAKE_OIL_OPR_DIFF",
                199,
                realization_index=realization_index,
            )

            assert len(data1.index) == 2000
            assert list(data1.columns) == [realization_index]

            realization_index = 150
            with pytest.raises(IndexError):
                data1 = GenDataCollector.loadGenData(
                    ert,
                    "default_0",
                    "SNAKE_OIL_OPR_DIFF",
                    199,
                    realization_index=realization_index,
                )
