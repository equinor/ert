#!/usr/bin/env python
#  Copyright (C) 2014  Statoil ASA, Norway. 
#   
#  The file 'test_fault_blocks.py' is part of ERT - Ensemble based Reservoir Tool.
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

from ert.ecl import EclGrid, EclTypeEnum , EclKW , EclRegion
from ert.test import ExtendedTestCase
from ert.ecl.faults import FaultBlock, FaultBlockLayer



class FaultBlockTest(ExtendedTestCase):
    def setUp(self):
        self.grid = EclGrid.create_rectangular( (10,10,10) , (1,1,1) )
        self.kw = EclKW.create( "FAULTBLK" , self.grid.size , EclTypeEnum.ECL_INT_TYPE )
        self.kw.assign( 1 )

        reg = EclRegion( self.grid , False )

        for k in range(self.grid.getNZ()):
            reg.clear( )
            reg.select_kslice( k , k )
            self.kw.assign( k , mask = reg )
            self.kw[ k * self.grid.getNX() * self.grid.getNY() + 7] = 177
    

    def test_fault_block(self):
        fault_block = FaultBlock( self.grid , 77 )
        self.assertEqual( 77 , fault_block.getBlockID() )
        
        fault_block.addCell( 0 )
        (x0,y0,z0) = self.grid.get_xyz( global_index = 0 )
        (xc,yc) = fault_block.getCentroid()
        
        self.assertFloatEqual( x0 , xc )
        self.assertFloatEqual( y0 , yc )




    def test_fault_block_layer(self):
        with self.assertRaises(ValueError):
            layer = FaultBlockLayer( self.grid , self.kw , -1 )

        with self.assertRaises(ValueError):
            layer = FaultBlockLayer( self.grid , self.kw , self.grid.size  )
            
        with self.assertRaises(ValueError):
            layer = FaultBlockLayer( self.grid , EclKW.create( "FAULTBLK" , 1 , EclTypeEnum.ECL_INT_TYPE ) , 0)

        with self.assertRaises(ValueError):
            layer = FaultBlockLayer( self.grid , EclKW.create( "FAULTBLK" , self.grid.size , EclTypeEnum.ECL_FLOAT_TYPE ) , 0)

        layer = FaultBlockLayer( self.grid , self.kw , 1 )
        self.assertEqual( 2 , layer.size() )
        self.assertEqual( 2 , len(layer) )

        with self.assertRaises(TypeError):
            ls = layer["JJ"]

        l = []
        for blk in layer:
            l.append( blk )
        self.assertEqual( len(l) , 2 )

        l0 = layer[0]
        l1 = layer[1]
        self.assertTrue( isinstance(l1 , FaultBlock ))

        with self.assertRaises(IndexError):
            l2 = layer[2]

            
        self.assertEqual( False , layer.hasBlock( 77 ))
        self.assertEqual( True , layer.hasBlock( 1 ))
        self.assertEqual( True , layer.hasBlock( 177 ))


        l1 = layer.getBlock( 1 )
        self.assertTrue( isinstance(l1 , FaultBlock ))
        
        l177 = layer.getBlock( 177 )
        with self.assertRaises(KeyError):
            l =layer.getBlock(66)

