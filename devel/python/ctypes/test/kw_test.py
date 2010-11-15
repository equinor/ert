#!/prog/sdpsoft/python2.4/bin/python
import math
import ert
import ert.ecl as ecl
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


