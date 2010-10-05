#!/prog/sdpsoft/python2.4/bin/python
import ert
import ert.ecl as ecl

grid_file = "data/eclipse/case/ECLIPSE.EGRID"
grid = ecl.EclGrid( grid_file )
print "Grid loaded"
