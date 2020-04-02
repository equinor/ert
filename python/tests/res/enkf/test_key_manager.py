from res.enkf.key_manager import KeyManager
from res.test import ErtTestContext

from tests import ResTest

class KeyManagerTest(ResTest):
    def setUp(self):
        self.config_file = self.createTestPath("local/snake_oil/snake_oil.ert")

    def test_summary_keys(self):
        with ErtTestContext("enkf_key_manager_test", self.config_file) as testContext:
            ert = testContext.getErt()
            key_man = KeyManager(ert)

            self.assertEqual( len(key_man.summaryKeys()), 47)
            self.assertTrue("FOPT" in key_man.summaryKeys())

            self.assertEqual(len(key_man.summaryKeysWithObservations()), 2)
            self.assertTrue("FOPR" in key_man.summaryKeysWithObservations())
            self.assertTrue(key_man.isKeyWithObservations("FOPR"))


    def test_gen_data_keys(self):
        with ErtTestContext("enkf_key_manager_test", self.config_file) as testContext:
            ert = testContext.getErt()
            key_man = KeyManager(ert)

            self.assertEqual( len(key_man.genDataKeys()), 3)
            self.assertTrue("SNAKE_OIL_WPR_DIFF@199" in key_man.genDataKeys())

            self.assertEqual(len(key_man.genDataKeysWithObservations()), 1)
            self.assertTrue("SNAKE_OIL_WPR_DIFF@199" in key_man.genDataKeysWithObservations())
            self.assertTrue(key_man.isKeyWithObservations("SNAKE_OIL_WPR_DIFF@199"))

    def test_custom_keys(self):
        with ErtTestContext("enkf_key_manager_test", self.config_file) as testContext:
            ert = testContext.getErt()
            key_man = KeyManager(ert)

            self.assertEqual( len(key_man.customKwKeys()), 2)
            self.assertTrue("SNAKE_OIL_NPV:NPV" in key_man.customKwKeys())

    def test_gen_kw_keys(self):
        with ErtTestContext("enkf_key_manager_test", self.config_file) as testContext:
            ert = testContext.getErt()
            key_man = KeyManager(ert)

            self.assertEqual(len(key_man.genKwKeys()), 10)
            self.assertTrue("SNAKE_OIL_PARAM:BPR_555_PERSISTENCE" in key_man.genKwKeys())

    def test_gen_kw_priors(self):
        with ErtTestContext("enkf_key_manager_test", self.config_file) as testContext:
            ert = testContext.getErt()
            key_man = KeyManager(ert)
            priors = key_man.gen_kw_priors()
            self.assertEqual(len(priors["SNAKE_OIL_PARAM"]), 10)
            self.assertTrue(
                {
                    "key" : "OP1_PERSISTENCE",
                    "function" : "UNIFORM",
                    "parameters" : {"MIN" : 0.01, "MAX" : 0.4}
                } in priors["SNAKE_OIL_PARAM"]
            )
