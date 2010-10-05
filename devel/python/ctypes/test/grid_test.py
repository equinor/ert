#!/prog/sdpsoft/python2.4/bin/python
import ert
import ert.ecl as ecl

egrid_file = "data/eclipse/case/ECLIPSE.EGRID"
egrid = ecl.EclGrid( egrid_file )

print "Loading grid:"
grid_file = "data/eclipse/case/ECLIPSE.GRID"
grid = ecl.EclGrid( grid_file )

print "Grid loaded"
