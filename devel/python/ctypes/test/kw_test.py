#!/usr/bin/env python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'kw_test.py' is part of ERT - Ensemble based Reservoir Tool. 
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


import math
import ert
import ert.ecl.ecl as ecl
import sys

def cutoff( x , arg ):
    if x < arg:
        return 0
    else:
        return x


init_file = ecl.EclFile( "data/eclipse/case/ECLIPSE.INIT" )
permx     = init_file.iget_named_kw("PERMX" , 0 )
poro      = init_file.iget_named_kw("PORO" , 0 )
grid      = ecl.EclGrid( "data/eclipse/case/ECLIPSE.EGRID" )

poro3d = grid.create3D( poro   , default = -100)

print "max:%g" % poro.max
print "min:%g" % poro.min

mask1 = ecl.EclRegion( grid , False )
mask2 = ecl.EclRegion( grid , False )
mask1.select_less( poro , 0.15 )
mask2.select_more( poro , 0.30 ) 

mask3  = mask1.copy()
mask3 |= mask2


print mask1.active_size
print mask2.active_size
print mask3.active_size

poro.apply( cutoff , mask = mask1 , arg = 0.05)


poro.write_grdecl( open("/tmp/poro_cos.grdecl" , "w") )

poro.add( permx , mask = mask1)
poro.sub( 1 )
poro.mul( poro )
poro.assign( 14.0 )
poro.div( 7.0 )

new_p = poro / 7.0

poro.write_grdecl( open("/tmp/poro_cos.grdecl" , "w") )

init_file = ecl.EclFile( "data/eclipse/case/ECLIPSE.INIT" )
permx_kw = init_file.iget_named_kw( "PERMX" , 0 )
permx_new = ecl.EclKW.new( "PERMX" , 3 , ecl.ECL_FLOAT_TYPE )
permx_new[0] = 1
permx_new[1] = 2
permx_new[2] = 3

init_file.replace_kw( permx_kw , permx_new )
fortio = ecl.FortIO( "/tmp/init" , "w")
init_file.fwrite( fortio )
fortio.close()

