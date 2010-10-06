#!/prog/sdpsoft/python2.4/bin/python
import ert
import ert.ecl as ecl

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


egrid_file  = "data/eclipse/case/ECLIPSE.EGRID"
grid_file   = "data/eclipse/case/ECLIPSE.GRID"
grdecl_file = "data/eclipse/case/include/example_grid_sim.GRDECL"    

grid = load_egrid( egrid_file )
grid = load_grid( grid_file )
grid = load_grdecl( grdecl_file )

print "Thickness(10,11,12): %g" % grid.cell_dz( ijk=(10,11,12) )
