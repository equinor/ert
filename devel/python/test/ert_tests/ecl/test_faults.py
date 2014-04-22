#!/usr/bin/env python
#  Copyright (C) 2014  Statoil ASA, Norway. 
#   
#  The file 'test_faults.py' is part of ERT - Ensemble based Reservoir Tool.
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
try:
    from unittest2 import skipIf
except ImportError:
    from unittest import skipIf

import time
from ert.ecl.faults import FaultCollection, Fault, FaultLine, FaultSegment
from ert.ecl import EclGrid
from ert.test import ExtendedTestCase

class FaultTest(ExtendedTestCase):
    def setUp(self):
        self.faults1 = self.createTestPath("local/ECLIPSE/FAULTS/fault1.grdecl")
        self.faults2 = self.createTestPath("local/ECLIPSE/FAULTS/fault2.grdecl")
        self.grid = EclGrid.create_rectangular( (151,100,50) , (1,1,1))
        
        
    def test_empty_collection(self):
        faults = FaultCollection(self.grid)
        self.assertEqual(0 , len(faults))

        self.assertFalse( faults.hasFault("FX") )
        
        with self.assertRaises(TypeError):
            f = faults[ [] ]

        with self.assertRaises(KeyError):
            f = faults["FX"]

        with self.assertRaises(IndexError):
            f = faults[0]


    def test_splitLine(self):
        faults = FaultCollection(self.grid)
        with self.assertRaises(ValueError):
            # Not slash terminated
            t = faults.splitLine("'F1'             149  149     29   29      1   43    'Y'")

        with self.assertRaises(ValueError):
            # Not integer
            t = faults.splitLine("'F1'             149  149     29   29      1   43X    'Y' /")

        with self.assertRaises(ValueError):
            # Missing item
            t = faults.splitLine("'F1'             149     29   29      1   43    'Y' /")

        with self.assertRaises(ValueError):
            # Quote fuckup
            t = faults.splitLine("'F1             149     149 29   29      1   43    'X' /")


    def test_empty_fault( self ):
        f = Fault(self.grid , "NAME")
        self.assertEqual("NAME" , f.getName())
        
        with self.assertRaises(KeyError):
            g = f["Key"]

        with self.assertRaises(KeyError):
            g = f[0]

        
    def test_empty_faultLine(self):
        fl = FaultLine(self.grid , 10)
        self.assertEqual( 10 , fl.getK())
        self.assertEqual( 0 , len(fl) )
        
        with self.assertRaises(TypeError):
            f = fl[ "Test" ]

        with self.assertRaises(IndexError):
            f = fl[0]


    def test_faultLine(self):
        fl = FaultLine(self.grid , 10)
        S1 = FaultSegment(0 , 10)
        S2 = FaultSegment(10 , 20)
        fl.append( S1 )
        fl.append( S2 )
        fl.verify()
        S3 = FaultSegment(20 , 30)
        fl.append( S3 )
        fl.verify()
        #---
        fl = FaultLine(self.grid , 10)
        S1 = FaultSegment(0 , 10)
        S2 = FaultSegment(20 , 10)
        fl.append( S1 )
        fl.append( S2 )
        fl.verify()
        #---
        fl = FaultLine(self.grid , 10)
        S1 = FaultSegment(10 , 0)
        S2 = FaultSegment(20 , 10)
        fl.append( S1 )
        fl.append( S2 )
        fl.verify()
        #---
        fl = FaultLine(self.grid , 10)
        S1 = FaultSegment(10 , 0)
        S2 = FaultSegment(10 , 20)
        fl.append( S1 )
        fl.append( S2 )
        fl.verify()

        fl = FaultLine(self.grid , 10)
        S1 = FaultSegment(10 , 0)
        S2 = FaultSegment(10 , 20)
        fl.append( S1 )
        fl.append( S2 )
        S3 = FaultSegment(40 , 30)
        with self.assertRaises(AssertionError):
            fl.append(S3)

    def test_load(self):
        faults = FaultCollection(self.grid , self.faults1)
        self.assertEqual( 3 , len(faults))
        faults.load( self.faults2 )
        self.assertEqual( 7 , len(faults))
    
    
    def test_iter(self):
        faults = FaultCollection(self.grid , self.faults1 , self.faults2)
        self.assertEqual( 7 , len(faults))
        c = 0
        for f in faults:
            c += 1
        self.assertEqual( c , len(faults))
    
    
    
    def test_fault(self):
        f = Fault(self.grid , "NAME")
    
        with self.assertRaises(ValueError):
            # Invalid face
            f.addRecord( 10 , 10 , 11 , 11 , 1 , 43 , "F")
            
    
        with self.assertRaises(ValueError):
            # Invalid coordinates
            f.addRecord( -1 , 10 , 11 , 11 , 1 , 43 , "X")
    
        with self.assertRaises(ValueError):
            # Invalid coordinates
            f.addRecord( 10000 , 10 , 11 , 11 , 1 , 43 , "X")
    
        with self.assertRaises(ValueError):
            # Invalid coordinates
            f.addRecord( 10 , 9 , 11 , 11 , 1 , 43 , "X")
    
    
        with self.assertRaises(ValueError):
            # Invalid coordinates
            f.addRecord( 10 , 9 , 11 , 11 , 1 , 43 , "X")
    
        with self.assertRaises(ValueError):
            # Invalid coordinates/face combination
            f.addRecord( 10 , 11 , 11 , 11 , 1 , 43 , "X")
    
        with self.assertRaises(ValueError):
            # Invalid coordinates/face combination
            f.addRecord( 10 , 11 , 11 , 12 , 1 , 43 , "Y")
    
        f.addRecord(10 , 10 , 0 , 10 , 1 , 10 , "X")
    
    
    def test_segment(self ):
        s0 = FaultSegment(0 , 10)
        self.assertEqual(s0.getC1() , 0 )
        self.assertEqual(s0.getC2() , 10 )
    
        s0.swap()
        self.assertEqual(s0.getC1() , 10 )
        self.assertEqual(s0.getC2() , 0 )
        
    
    
    def test_fault_line(self ):
        faults = FaultCollection(self.grid , self.faults1 , self.faults2)
        for fault in faults:
            for layer in fault:
                for fl in layer:
                    fl.verify()

