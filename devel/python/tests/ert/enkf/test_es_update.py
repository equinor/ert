from ert.test import ExtendedTestCase, ErtTestContext
from ert.enkf import ESUpdate


class ESUpdateTest(ExtendedTestCase):

    def test_create(self):
        config = self.createTestPath("local/custom_kw/mini_config")
        with ErtTestContext("python/enkf/data/custom_kw_simulated", config) as context:
            ert = context.getErt()
            es_update = ESUpdate( ert )

            
            
