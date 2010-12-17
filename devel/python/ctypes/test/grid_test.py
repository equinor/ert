#!/prog/sdpsoft/python2.4/bin/python
import ert
import ert.ecl as ecl
from   ert.util.tvector import DoubleVector
from   ert.util.tvector import DoubleVector

def load_grid( grid_file ):
    grid = ecl.EclGrid( grid_file )
    return grid


def load_egrid( egrid_file ):
    grid = ecl.EclGrid( egrid_file )
    return grid


def load_grdecl( grdecl_file ):
    fileH = open( grdecl_file , "r")
    specgrid = ecl.EclKW.grdecl_load( fileH , "SPECGRID" , ecl_type = ecl.ECL_INT_TYPE )
    zcorn    = ecl.EclKW.grdecl_load( fileH , "ZCORN" )
    coord    = ecl.EclKW.grdecl_load( fileH , "COORD" )
    actnum   = ecl.EclKW.grdecl_load( fileH , "ACTNUM" , ecl_type = ecl.ECL_INT_TYPE )

    grid = ecl.EclGrid.create( specgrid , zcorn , coord , actnum )
    return grid


init_file   = ecl.EclFile( "data/eclipse/case/ECLIPSE.INIT" )
egrid_file  = "data/eclipse/case/ECLIPSE.EGRID"
grid_file   = "data/eclipse/case/ECLIPSE.GRID"
grdecl_file = "data/eclipse/case/include/example_grid_sim.GRDECL"    

grid = load_grdecl( grdecl_file )
grid = load_grid( grid_file )
grid = load_egrid( egrid_file )

print "Thickness(10,11,12): %g" % grid.cell_dz( ijk=(10,11,12) )

permx_column = DoubleVector( -999 )
grid.load_column( init_file.iget_named_kw( "PERMX" , 0 ) , 5 , 5 , permx_column)
permx_column.printf()

print "top2    : %g   depth(10,10,0)    : %g " % (grid.top( 10, 10) , grid.depth( ijk=(10,10,0)))
print "bottom2 : %g   depth(10,10,nz-1) : %g " % (grid.bottom( 10 , 10 ) , grid.depth( ijk=(10,10,grid.nz - 1)))

