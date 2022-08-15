import pytest

from ....libres_utils import ResTest

from ert._c_wrappers.enkf.export import GenKwCollector
from ert._c_wrappers.test import ErtTestContext


class GenKwCollectorTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def test_gen_kw_collector(self):
        with ErtTestContext(self.config) as context:
            ert = context.getErt()

            data = GenKwCollector.loadAllGenKwData(ert, "default_0")

            assert (
                pytest.approx(data["SNAKE_OIL_PARAM:OP1_PERSISTENCE"][0], rel=1e-5)
                == 0.047517
            )
            assert (
                pytest.approx(data["SNAKE_OIL_PARAM:OP1_PERSISTENCE"][24], rel=1e-5)
                == 0.160907
            )

            assert (
                pytest.approx(data["SNAKE_OIL_PARAM:OP1_OFFSET"][0], rel=1e-5)
                == 0.054539
            )
            assert (
                pytest.approx(data["SNAKE_OIL_PARAM:OP1_OFFSET"][12], rel=1e-5)
                == 0.057807
            )

            # pylint: disable=pointless-statement
            # realization 20:
            data.loc[20]

            with self.assertRaises(KeyError):
                # realization 60:
                data.loc[60]

            data = GenKwCollector.loadAllGenKwData(
                ert,
                "default_0",
                ["SNAKE_OIL_PARAM:OP1_PERSISTENCE", "SNAKE_OIL_PARAM:OP1_OFFSET"],
            )

            assert (
                pytest.approx(data["SNAKE_OIL_PARAM:OP1_PERSISTENCE"][0], rel=1e-5)
                == 0.047517
            )
            assert (
                pytest.approx(data["SNAKE_OIL_PARAM:OP1_OFFSET"][0], rel=1e-5)
                == 0.054539
            )

            with self.assertRaises(KeyError):
                data["SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE"]

            realization_index = 10
            data = GenKwCollector.loadAllGenKwData(
                ert,
                "default_0",
                ["SNAKE_OIL_PARAM:OP1_PERSISTENCE"],
                realization_index=realization_index,
            )

            assert data.index == [realization_index]
            assert len(data.index) == 1
            assert list(data.columns) == ["SNAKE_OIL_PARAM:OP1_PERSISTENCE"]
            assert (
                pytest.approx(data["SNAKE_OIL_PARAM:OP1_PERSISTENCE"][10], rel=1e-5)
                == 0.282923
            )

            non_existing_realization_index = 150
            with pytest.raises(IndexError):
                data = GenKwCollector.loadAllGenKwData(
                    ert,
                    "default_0",
                    ["SNAKE_OIL_PARAM:OP1_PERSISTENCE"],
                    realization_index=non_existing_realization_index,
                )
