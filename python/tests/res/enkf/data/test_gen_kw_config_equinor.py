import pytest

from tests import ResTest
from res.test import ErtTestContext

from res.enkf import ErtImplType, GenKwConfig

@pytest.mark.equinor_test
class GenKwConfigTest(ResTest):

    def setUp(self):
        self.config = self.createTestPath("Equinor/config/with_data/config")


    def test_gen_kw_config(self):

        with ErtTestContext("python/enkf/data/gen_kw_config", self.config) as context:

            ert = context.getErt()

            result_gen_kw_keys = ert.ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_KW)

            expected_keys = ["GRID_PARAMS", "FLUID_PARAMS", "MULTFLT"]

            self.assertItemsEqual(result_gen_kw_keys, expected_keys)
            self.assertEqual(len(expected_keys), len(result_gen_kw_keys))

            for key in expected_keys:
                node = ert.ensembleConfig().getNode(key)
                gen_kw_config = node.getModelConfig()
                self.assertIsInstance(gen_kw_config, GenKwConfig)

                self.assertEqual(gen_kw_config.getKey(), key)

                if key == "GRID_PARAMS":
                    expected_values = ["MULTPV2", "MULTPV3"]

                    self.assertFalse(gen_kw_config.shouldUseLogScale(0))
                    self.assertFalse(gen_kw_config.shouldUseLogScale(1))

                elif key == "MULTFLT":
                    expected_values = ["F3"]

                    self.assertTrue(gen_kw_config.shouldUseLogScale(0))

                elif key == "FLUID_PARAMS":
                    expected_values = ["SWCR", "SGCR"]
                    self.assertFalse(gen_kw_config.shouldUseLogScale(0))
                    self.assertFalse(gen_kw_config.shouldUseLogScale(1))

                self.assertEqual([value for value in gen_kw_config], expected_values)