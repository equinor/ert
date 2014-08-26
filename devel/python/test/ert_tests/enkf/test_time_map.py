import datetime 

from ert.enkf.enums.realization_state_enum import RealizationStateEnum
from ert.enkf import TimeMap
from ert.test import TestAreaContext
from ert.test import ExtendedTestCase


class TimeMapTest(ExtendedTestCase):

    def test_time_map(self):
        with self.assertRaises(IOError):
            TimeMap("Does/not/exist")

    
        tm = TimeMap()
        with self.assertRaises(IndexError):
            t = tm[10]
            
        self.assertTrue( tm.update(0 , datetime.date(2000 , 1, 1)))
        self.assertEqual( tm[0] , datetime.date(2000 , 1, 1))
        
        self.assertTrue( tm.isStrict() )
        with self.assertRaises(Exception):
            tm.update(tm.update(0 , datetime.date(2000 , 1, 2)))

        tm.setStrict( False )
        self.assertFalse(tm.update(0 , datetime.date(2000 , 1, 2)))

        tm.setStrict( True )
        self.assertTrue( tm.update( 1 , datetime.date(2000 , 1, 2)))
        d = tm.dump()
        self.assertEqual( d , [(0 , datetime.date(2000,1,1) , 0),
                               (1 , datetime.date(2000,1,2) , 1)])
        
