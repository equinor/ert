import os
from ert.enkf import EnkfFs
from ert.enkf import EnKFMain
from ert.enkf import ErtTestContext
from ert_tests import ExtendedTestCase


class EnKFFSTest(ExtendedTestCase):
    def setUp(self):
        self.mount_point = self.createTestPath("Statoil/config/with_data/storage/default")
        self.config_file = self.createTestPath("Statoil/config/with_data/config")

    def test_create(self):
        print self.mount_point
        fs = EnkfFs( self.mount_point )
        self.assertEqual( 1 , fs.refCount() ) 

        
    def test_throws(self):
        with self.assertRaises(Exception):
            fs = EnkfFs("/does/not/exist")
            
    
    def test_refcount(self):
        with ErtTestContext("TEST" , self.config_file) as testContext:
            ert = testContext.getErt()
            self.assertTrue( isinstance( ert , EnKFMain ))

            fsm = ert.getEnkfFsManager()
            fs = fsm.getCurrentFS()
            self.assertEqual( 2 , fs.refCount())

            

