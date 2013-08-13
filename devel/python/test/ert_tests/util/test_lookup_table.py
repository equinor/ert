from unittest2 import TestCase
from ert.util import LookupTable
import numpy

__author__ = 'jpb'

class TestLookupTable(TestCase):

    def test_lookup_table(self):
        lookup = LookupTable()

        self.assertTrue(numpy.isnan(lookup.max))
        self.assertTrue(numpy.isnan(lookup.min))
        self.assertTrue(numpy.isnan(lookup.arg_max))
        self.assertTrue(numpy.isnan(lookup.arg_min))
        self.assertEqual(len(lookup), 0)


        lookup.append(0.0, 0.0)
        lookup.append(1.0, 1.0)

        self.assertEqual(lookup.max, 1.0)
        self.assertEqual(lookup.min, 1.0)
        self.assertEqual(lookup.arg_max, 1.0)
        self.assertEqual(lookup.arg_min, 0.0)
        self.assertEqual(len(lookup), 2)

        self.assertEqual(lookup.interp(0.5), 0.5)

