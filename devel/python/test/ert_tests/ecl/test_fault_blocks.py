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
from ert.ecl.faults import FaultBlock, FaultBlockLayer, FaultBlockCollection, FaultBlockCell

def create_FaultBlock():
    grid = EclGrid.create_rectangular( (10,10,10) , (1,1,1) )
    return FaultBlock(grid , 0 , 0)


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
        fault_block = FaultBlock( self.grid , 0 , 77 )
        self.assertEqual( 77 , fault_block.getBlockID() )
        
        fault_block.addCell( 0 , 0)
        (x0,y0,z0) = self.grid.get_xyz( global_index = 0 )
        (xc,yc) = fault_block.getCentroid()
        
        self.assertFloatEqual( x0 , xc )
        self.assertFloatEqual( y0 , yc )
        fault_block.addCell( 1 , 1)
        fault_block.addCell( 2 , 2)

        self.assertEqual(len(fault_block) , 3)

        g_list = fault_block.getGlobalIndexList()
        self.assertEqual(len(g_list) , 3)
        self.assertEqual( g_list[0] , 0 )
        self.assertEqual( g_list[1] , 1 + self.grid.getNX() )
        self.assertEqual( g_list[2] , 2*(1 + self.grid.getNX()) )
        
        g_list[0] = 1000
        g_list2 = fault_block.getGlobalIndexList()
        self.assertEqual( g_list2[0] , 0 )
        
        self.assertTrue( isinstance(fault_block[0] , FaultBlockCell)  )

        cell = fault_block[0]
        self.assertEqual( cell.i , 0 )
        self.assertEqual( cell.j , 0 )
        self.assertEqual( cell.k , 0 )

        self.assertEqual( cell.x , x0 )
        self.assertEqual( cell.y , y0 )
        self.assertEqual( cell.z , z0 )

        with self.assertRaises(TypeError):
            c = fault_block["STRING"]

        with self.assertRaises(IndexError):
            c = fault_block[len(fault_block)]
            
        self.assertEqual( fault_block[-1].x , fault_block[len(fault_block) - 1].x)
        
        
    def test_fault_block_assign2region(self):
        fault_block = create_FaultBlock()
        fault_block.assignToRegion( 2 )
        self.assertEqual( [2] , list(fault_block.getRegionList()))

        fault_block.assignToRegion( 2 )
        self.assertEqual( [2] , list(fault_block.getRegionList()))

        fault_block.assignToRegion( 3 )
        self.assertEqual( [2,3] , list(fault_block.getRegionList()))

        fault_block.assignToRegion( 1 )
        self.assertEqual( [1,2,3] , list(fault_block.getRegionList()))

        fault_block.assignToRegion( 2 )
        self.assertEqual( [1,2,3] , list(fault_block.getRegionList()))

       
 

    def test_fault_block_gc(self):
        fault_block = create_FaultBlock()
        fault_block.addCell( 0 , 0)
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

            
        self.assertEqual( False , 77 in layer)
        self.assertEqual( True , 1 in layer)
        self.assertEqual( True , 177 in layer)


        l1 = layer.getBlock( 1 )
        self.assertTrue( isinstance(l1 , FaultBlock ))
        
        l177 = layer.getBlock( 177 )
        with self.assertRaises(KeyError):
            l =layer.getBlock(66)

        with self.assertRaises(KeyError):
            layer.deleteBlock(66)

        layer.deleteBlock(177)
        self.assertEqual( 1 , len(layer))
        blk = layer[0]
        self.assertEqual( blk.getBlockID() , 1 ) 

        with self.assertRaises(KeyError):
            layer.addBlock(1)
            
        blk2 = layer.addBlock(2)
        self.assertEqual( len(layer) , 2 ) 
        
        layer.addBlock(100)
        layer.addBlock(101)
        layer.addBlock(102)
        layer.addBlock(103)

        layer.deleteBlock(2)
        blk1 = layer.getBlock( 103 )
        blk2 = layer[-1]
        self.assertEqual( blk1.getBlockID() , blk2.getBlockID() )




    def test_fault_block_collection(self):
        collection = FaultBlockCollection( self.grid , self.kw )
        self.assertTrue( len(collection) , self.grid.getNZ() )

        layer_list = []
        for layer in collection:
            layer_list.append( layer )
        
        for k in range(self.grid.getNZ()):
            self.assertEqual( collection[k], collection.getLayer(k) )

        self.assertTrue( len(layer_list) , self.grid.getNZ() )
        
