import os
from ert.enkf import EnkfFs
from ert.enkf import EnKFMain
from ert.enkf import EnkfFsManager
from ert.enkf import ErtTestContext
from ert_tests import ExtendedTestCase


class EnKFFSManagerTest2(ExtendedTestCase):
    def setUp(self):
        self.config_file = self.createTestPath("Statoil/config/with_data/config")
            
    

    def test_rotate(self):
        # We are indirectly testing the create through the create
        # already in the enkf_main object. In principle we could
        # create a separate manager instance from the ground up, but
        # then the reference count will be weird.
        with ErtTestContext("TEST" , self.config_file) as testContext:
            ert = testContext.getErt()
            fsm = ert.getEnkfFsManager()
            self.assertEqual( 1 , fsm.size() )

            fs1 = fsm.getFS("FSA")
            fs2 = fsm.getFS("FSB")
            self.assertEqual( fsm.capacity , fsm.size() )

            fsList = []
            for i in range(10):
                fs = "FS%d" % i
                print "Mounting:%s" % fs
                fsList.append( fsm.getFS(fs))
                self.assertEqual( fsm.capacity , fsm.size() )

            
