#!/usr/bin/env python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'sum_test.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details. 

import os
import datetime

try:
    from unittest2 import skipIf, skipUnless, skipIf
except ImportError:
    from unittest import skipIf, skipUnless, skipIf

from ert.ecl import EclSum
from ert.ecl import EclNPV

from ert.util import StringList, TimeVector, DoubleVector
from ert.util.test_area import TestAreaContext

from ert_tests import ExtendedTestCase


base = "ECLIPSE"
path = "Statoil/ECLIPSE/Gurbat"
case = "%s/%s" % (path, base)


class NPVTest(ExtendedTestCase):
    def setUp(self):
        self.case = self.createTestPath(case)


    def test_create(self):
        with self.assertRaises(Exception):
            npv = EclNPV("/does/not/exist")

        npv = EclNPV( self.case )


    def test_eval_npv(self):
        npv = EclNPV( self.case )
        with self.assertRaises(ValueError):
            npv.eval()

            
    def test_expression(self):
        npv = EclNPV( self.case )
        self.assertIsNone( npv.getExpression() )
        npv.setExpression( "[FOPT]*$OIL_PRICE - [FGIT]*$GAS_PRICE")
        self.assertEqual( npv.getExpression() , "[FOPT]*$OIL_PRICE - [FGIT]*$GAS_PRICE")
        self.assertIn( "FOPT" , npv.getKeyList() )
        self.assertIn( "FGIT" , npv.getKeyList() )


        with self.assertRaises(ValueError):
            npv.setExpression("[FOPT")

        with self.assertRaises(ValueError):
            npv.setExpression("FOPT]")

        with self.assertRaises(KeyError):
            npv.setExpression("[FoPT]")
