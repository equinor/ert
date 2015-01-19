import datetime 

from ert.util import BoolVector
from ert.test import TestAreaContext
from ert.test import ExtendedTestCase
from ert.enkf import MeasBlock,MeasData



class MeasDataTest(ExtendedTestCase):


    def test_create(self):
        ens_size = 10
        ens_mask = BoolVector( default_value = True , initial_size = ens_size )
        data = MeasData( ens_mask )
        self.assertTrue( isinstance( data , MeasData ))
        
        block1 = data.addBlock( "OBS1" , 10 , 5 )
        block2 = data.addBlock( "OBS2" , 10 , 10 )

        self.assertTrue( isinstance( block1 , MeasBlock ))
        self.assertTrue( isinstance( block2 , MeasBlock ))
        
        self.assertEqual( block1.getObsSize() , 5 )
        self.assertEqual( block2.getObsSize() , 10 )
        
        with self.assertRaises(ValueError):
            S = data.createS()

        for iens in range(ens_size):
            block1[0,iens] = 5
            block2[0,iens] = 10
            block2[1,iens] = 15

        self.assertEqual( 3 , data.activeObsSize() )
        S = data.createS()

        self.assertEqual( S.dims() , (3 , ens_size) )
        
        for iens in range(ens_size):
            self.assertEqual( S[0,iens] , 5 )
            self.assertEqual( S[1,iens] , 10 )
            self.assertEqual( S[2,iens] , 15 )
            

    

