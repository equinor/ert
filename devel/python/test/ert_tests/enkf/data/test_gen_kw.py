from ert.cwrap import clib, CWrapper
from ert.enkf.data import GenKw
from ert.enkf.data import GenKwConfig
from ert.test import ErtTestContext
from ert.test.extended_testcase import ExtendedTestCase

test_lib  = clib.ert_load("libenkf")
cwrapper =  CWrapper(test_lib)


class GenKwTest(ExtendedTestCase):
    def setUp(self):
        self.config_file = self.createTestPath("Statoil/config/forward/config_GEN_KW_true")

    def test_gen_kw_get_set(self):
        with ErtTestContext("gen_kw_test", self.config_file) as test_context:
            ert = test_context.getErt()

            enkf_config_node = ert.ensembleConfig().getNode("MULTFLT")
            gen_kw_config = enkf_config_node.getModelConfig()
            self.assertIsInstance(gen_kw_config, GenKwConfig)

            gen_kw = GenKw(gen_kw_config)
            self.assertIsInstance(gen_kw, GenKw)

            gen_kw[0] = 3.0
            self.assertEqual(gen_kw[0], 3.0)

            gen_kw["MULTFLT"] = 4.0
            self.assertEqual(gen_kw["MULTFLT"], 4.0)
            self.assertEqual(gen_kw[0], 4.0)

































