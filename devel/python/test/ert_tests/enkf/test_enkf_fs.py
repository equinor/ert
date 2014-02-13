import os
from ert.enkf.enkf_fs import EnkfFs
from ert_tests import ExtendedTestCase

class EnKFFSTest(ExtendedTestCase):
    def setUp(self):
        self.mount_point = self.createTestPath("Statoil/config/with_data/storage/default")
        

    def test_create(self):
        print self.mount_point
        fs = EnkfFs( self.mount_point )
        self.assertEqual( 1 , fs.refCount() ) 

        
    def test_throws(self):
        with self.assertRaises(Exception):
            fs = EnkfFs("/does/not/exist")
            



