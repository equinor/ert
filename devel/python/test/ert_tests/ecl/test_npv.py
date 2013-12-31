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
import math

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
            npv.parseExpression("[FOPT")

        with self.assertRaises(ValueError):
            npv.parseExpression("FOPT]")

        with self.assertRaises(KeyError):
            npv.parseExpression("[FoPT]")
            
        with self.assertRaises(ValueError):
            npv.parseExpression("[FOPR]")
            
        parsedExpression = npv.parseExpression("[FOPT]")
        self.assertEqual( parsedExpression , "FOPT[i]")
        self.assertEqual( 1 , len(npv.getKeyList() ))


        parsedExpression = npv.parseExpression("[FOPT]*2 + [FGPT] - [WOPT:OP_1]")
        self.assertEqual( parsedExpression , "FOPT[i]*2 + FGPT[i] - WOPT_OP_1[i]")
        keyList = npv.getKeyList()
        self.assertEqual( 3 , len(keyList))
        self.assertIn( "FOPT" , keyList )
        self.assertIn( "FGPT" , keyList )
        self.assertIn( "WOPT:OP_1" , keyList )


    def test_period(self):
        npv = EclNPV( self.case )
        self.assertIsNone(npv.start)
        self.assertIsNone(npv.end)
        self.assertEqual("1Y" , npv.interval)


    def test_eval(self):
        npv = EclNPV(self.case)
        npv.compile("[FOPT]")
        npv1 = npv.evalNPV()

        npv2 = 0
        sum = EclSum(self.case)
        trange = sum.timeRange()
        fopr = sum.blockedProduction("FOPT" , trange)
        for v in fopr:
            npv2 += v
        self.assertAlmostEqual( npv1 , npv2 )
        
        npv.compile("[FOPT] - 0.5*[FOPT] - 0.5*[FOPT]")
        npv1 = npv.evalNPV()
        self.assertTrue( abs(npv1) < 1e-2 )

        npv.compile("[WOPT:OP_1] - 0.5*[WOPT:OP_1] - 0.5*[WOPT:OP_1]")
        npv1 = npv.evalNPV()
        self.assertTrue( abs(npv1) < 1e-2 )
