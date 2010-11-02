#!/prog/sdpsoft/python2.4/bin/python
import datetime
import ert
import ert.ecl as ecl

case = "data/eclipse/case/ECLIPSE"
sum  = ecl.EclSum( case ) 

print sum.get_interp( "WWCT:OP_3" , days = 750 )
print sum.get_interp( "WWCT:OP_3" , date = datetime.date( 2004 , 1, 1 ))
