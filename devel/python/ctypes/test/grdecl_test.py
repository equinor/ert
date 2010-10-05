#!/prog/sdpsoft/python2.4/bin/python
import ert
import ert.ecl as ecl

file = "data/eclipse/case/include/example_grid_sim.GRDECL"
fileH = open( file , "r")
specgrid = ecl.EclKW.grdecl_load( fileH , "SPECGRID" , ecl_type = ecl.ECL_INT_TYPE )
zcorn    = ecl.EclKW.grdecl_load( fileH , "ZCORN" )
coord    = ecl.EclKW.grdecl_load( fileH , "COORD" )
actnum   = ecl.EclKW.grdecl_load( fileH , "ACTNUM" , ecl_type = ecl.ECL_INT_TYPE )

print "Creating grid ...."
grid = ecl.EclGrid.create( specgrid , zcorn , coord , actnum )
print  grid.size
print  grid.nx
print  grid.nactive
