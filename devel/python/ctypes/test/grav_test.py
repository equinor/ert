#!/prog/sdpsoft/python2.4/bin/python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'grav_test.py' is part of ERT - Ensemble based Reservoir Tool. 
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


import datetime
import ert
import ert.ecl as ecl

pos = (5354 , 9329 , 100)
grid     = ecl.EclGrid("data/eclipse/case/ECLIPSE.EGRID")
init     = ecl.EclFile("data/eclipse/case/ECLIPSE.INIT")
restart1 = ecl.EclFile.restart_block("data/eclipse/case/ECLIPSE.UNRST" , report_step = 10)
restart2 = ecl.EclFile.restart_block("data/eclipse/case/ECLIPSE.UNRST" , report_step = 40)



# Troll in Bergen
# pos      = (530991 , 6754822 , 342.785)
# grid     = ecl.EclGrid("/d/proj/bg/restroll2/restek2/TEG/simu_HM2009/BCUPD_HISTORYMATCH_JAN10_059.EGRID")
# init     = ecl.EclFile("/d/proj/bg/restroll2/restek2/TEG/simu_HM2009/BCUPD_HISTORYMATCH_JAN10_059.INIT")
# restart1 = ecl.EclFile.restart_block("/d/proj/bg/restroll2/restek2/TEG/simu_HM2009/BCUPD_HISTORYMATCH_JAN10_059.UNRST" , report_step = 10)
# restart2 = ecl.EclFile.restart_block("/d/proj/bg/restroll2/restek2/TEG/simu_HM2009/BCUPD_HISTORYMATCH_JAN10_059.UNRST" , report_step = 40)

if restart1 and restart2:
    deltaG = ecl.ecl_grav.deltag( pos , grid , init , restart1 , restart2 )

    print deltaG
else:
    print "Load failed"
