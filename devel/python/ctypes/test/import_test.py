#!/usr/bin/env python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'import_test.py' is part of ERT - Ensemble based Reservoir Tool. 
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





print  "import ert" 
import ert


print "import ert.ecl"
from ert.ecl.ecl import *
g = EclGrid( "Bjarne" ) 


print "import ert.job_queue"
import ert.job_queue 


print "import ert.util"
import ert.util.stringlist
import ert.util.tvector
import ert.util.stat

#import ert.ecl.libecl
#import ert.config.libconfig
#import ert.rms.librms
#import ert.sched.libsched
#import ert.enkf.libenkf


