#!/prog/sdpsoft/python2.4/bin/python
import ert
import ert.ecl as ecl


init_file = ecl.EclFile( "data/eclipse/case/ECLIPSE.INIT" )
permx     = init_file.iget_named_kw("PERMX" , 0 )
poro      = init_file.iget_named_kw("PORO" , 0 )
grid      = ecl.EclGrid( "data/eclipse/case/ECLIPSE.EGRID" )

mask1 = ecl.EclRegion( grid , False )
mask2 = ecl.EclRegion( grid , False )
mask1.select_less( poro , 0.15 )
mask2.select_more( poro , 0.30) 

mask3  = mask1.copy()
mask3 |= mask2

print mask1.active_size
print mask2.active_size
print mask3.active_size
