#!/prog/sdpsoft/python2.4/bin/python
import ert
import ert.ecl as ecl

case = "data/eclipse/case/ECLIPSE"
sum  = ecl.EclSum( case ) 

print sum.get_interp( "WWCT:OP_3" , days = 750 )
