from datetime import datetime,date
from ert.util import ctime

try:
    from unittest2 import TestCase
except ImportError:
    from unittest import TestCase


class CTimeTest(TestCase):

    def test_c_time(self):
        c_time = ctime(0)
        self.assertEqual(str(c_time), "1970-01-01 01:00:00")

        date_time = ctime(datetime(1970, 1, 1, 1, 0, 0))
        self.assertEqual(c_time, date_time)

        date_time_after = ctime(datetime(1970, 1, 1, 1, 0, 5))

        self.assertTrue(date_time_after > date_time)


    def test_math(self):
        c1 = ctime( date(2000 , 1 , 1 ))
        c2 = ctime( date(2000 , 1 , 1 ))
        c3 = ctime( date(2000 , 1 , 1 ))

        c3 += c1
        self.assertTrue( isinstance(c3 , ctime))

        c4 = c1 * 1
        c4 = c1 * 1.0
        self.assertTrue( isinstance(c4 , ctime))
        self.assertTrue( isinstance(c1 + c2  , ctime))
        
        self.assertEqual( (c1 + c2)*0.5 , date(2000 , 1 , 1))
        
