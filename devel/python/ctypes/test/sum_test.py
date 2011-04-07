#!/usr/bin/env python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'sum_test.py' is part of ERT - Ensemble based Reservoir Tool. 
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

case = "data/eclipse/case/ECLIPSE"
sum  = ecl.EclSum( case ) 

print sum.get_interp( "WWCT:OP_3" , days = 750 )
print sum.get_interp( "WWCT:OP_3" , date = datetime.date( 2004 , 1, 1 ))
