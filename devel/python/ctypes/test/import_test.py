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


import sys
sys.path = ["/tmp"] + sys.path
import ctypes
print sys.path

print  "import ert", 
sys.stdout.flush()
import ert
print

print "import ert.ecl",
sys.stdout.flush()
import ert.ecl
print

print "import ert.job_queue",
sys.stdout.flush()
import ert.job_queue 
print

print "import ert.util",
sys.stdout.flush()
import ert.util.stringlist
import ert.util.tvector
import ert.util.stat
print
