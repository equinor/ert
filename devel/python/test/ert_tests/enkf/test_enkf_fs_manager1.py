import os
from ert.enkf import EnkfFs
from ert.enkf import EnKFMain
from ert.enkf import EnkfFsManager
from ert.enkf import ErtTestContext
from ert_tests import ExtendedTestCase


class EnKFFSManagerTest1(ExtendedTestCase):
    def setUp(self):
        self.config_file = self.createTestPath("Statoil/config/with_data/config")
            

    
    def test_create(self):
        # We are indirectly testing the create through the create
        # already in the enkf_main object. In principle we could
        # create a separate manager instance from the ground up, but
        # then the reference count will be weird.
        with ErtTestContext("TEST" , self.config_file) as testContext:
            ert = testContext.getErt()
            fsm = ert.getEnkfFsManager()
            
            fs = fsm.getCurrentFS( )
            self.assertEqual( 2 , fs.refCount())
            self.assertEqual( 1 , fsm.size() )
            
            fs2 = fsm.getFS( "newFS" )
            self.assertEqual( 2 , fsm.size() )
            self.assertEqual( 1 , fs2.refCount())
            
            with self.assertRaises(IOError):
                fs2 = fsm.getFS( "newFS3" , read_only = True)

