import datetime 

from ert.test import TestAreaContext
from ert.test import ExtendedTestCase
from ert.enkf import MeasBlock



class MeasBlockTest(ExtendedTestCase):


    def test_create(self):
        key = "OBS"
        ens_size = 100
        obs_size = 77
        block = MeasBlock( key , obs_size , ens_size)
        self.assertEqual( block.getObsSize() , obs_size )
        self.assertEqual( block.getEnsSize() , ens_size )



    def test_update(self):
        key = "OBS"
        obs_size = 4
        ens_size = 10
        block = MeasBlock( key , obs_size , ens_size)

        with self.assertRaises(TypeError):
            block["String"] = 10

        with self.assertRaises(TypeError):
            block[10] = 10

        with self.assertRaises(IndexError):
            block[obs_size,0] = 10

        with self.assertRaises(IndexError):
            block[0,ens_size] = 10
            
        #-----------------------------------------------------------------

        with self.assertRaises(TypeError):
            a = block["String"]

        with self.assertRaises(TypeError):
            a = block[10]

        with self.assertRaises(IndexError):
            val = block[obs_size,0]

        with self.assertRaises(IndexError):
            val = block[0,ens_size]

        block[1,2] = 3
        self.assertEqual( 3 , block[1,2] )

            

    def test_stat(self):
        key = "OBS"
        obs_size = 2
        ens_size = 10
        block = MeasBlock( key , obs_size , ens_size)

        for iens in range(ens_size):
            block[0,iens] = iens
            block[1,iens] = iens + 1
        
        self.assertEqual( 4.5 , block.igetMean( 0 ))
        self.assertEqual( 5.5 , block.igetMean( 1 ))
        
        self.assertFloatEqual( 2.872281 , block.igetStd( 0 ))
        self.assertFloatEqual( 2.872281 , block.igetStd( 1 ))

        

