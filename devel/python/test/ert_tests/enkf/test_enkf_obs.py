import os
from ert.enkf import EnkfFs
from ert.enkf import ObsVector
from ert.enkf import EnKFMain
from ert.enkf import EnkfFsManager
from ert.enkf import ErtTestContext
from ert.util import BoolVector
from ert_tests import ExtendedTestCase

class EnKFObsTest(ExtendedTestCase):
    def setUp(self):
        self.config_file = self.createTestPath("Statoil/config/obs_testing/config")

    def testObs(self):
        with ErtTestContext("obs_test" , self.config_file) as testContext:
            ert = testContext.getErt()
            obs = ert.getObservations()

            self.assertEqual(31 , len(obs))
            for v in obs:
                self.assertTrue( isinstance(v , ObsVector ))
                
            with self.assertRaises(ValueError):
                v = obs[-1]

            with self.assertRaises(IndexError):
                v = obs[40]
                
            with self.assertRaises(KeyError):
                v = obs["No-this-does-not-exist"]
            
            v1 = obs["WWCT:OP_3"]
            v2 = obs["GOPT:OP"]
            mask = BoolVector(True , ert.getEnsembleSize() )
            currentFS = ert.getEnkfFsManager().getCurrentFileSystem()

            self.assertTrue( v1.hasData( mask , currentFS ))
            self.assertFalse( v2.hasData( mask , currentFS))
            
