#!/usr/bin/env python
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
import ert.ecl.ecl as ecl
import sys

restart1  = ecl.EclFile.restart_block("data/eclipse/grav/TROLL.UNRST" , report_step = 117 )
restart2  = ecl.EclFile.restart_block("data/eclipse/grav/TROLL.UNRST" , report_step = 199 )
grid_file = "data/eclipse/grav/TROLL.EGRID"
init_file = "data/eclipse/grav/TROLL.INIT"

rporv1 = restart1.iget_named_kw( "RPORV" , 0 ) 
rporv2 = restart2.iget_named_kw( "RPORV" , 0 )

pormod1 = restart1.iget_named_kw( "PORV_MOD" , 0 ) 
pormod2 = restart2.iget_named_kw( "PORV_MOD" , 0 )

for i in range( rporv1.size ):
    print "%d  %g  %g  %g  %g" % (i , rporv1[i] , rporv2[i], pormod1[i] , pormod2[i])


grav = ecl.EclGrav( grid_file , init_file )
grav.add_survey_PORMOD("BASE"    , restart1 )
grav.add_survey_PORMOD("MONITOR" , restart2 )

print " %g " % grav.eval( "BASE" , "MONITOR" , 541003 , 6709907 , 297.023)
