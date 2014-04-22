import os
from ert.enkf import EnkfFs
from ert.enkf import EnKFMain
from ert.enkf import EnkfFsManager
from ert.enkf import ErtTestContext
from ert.test import ExtendedTestCase


class ErtTestContextTest(ExtendedTestCase):
    def setUp(self):
        self.config_file = self.createTestPath("Statoil/config/obs_data/config")

    def testRaises(self):
        with self.assertRaises(IOError):
            testContext = ErtTestContext("ExistTest" , "Does/not/exist")

            
