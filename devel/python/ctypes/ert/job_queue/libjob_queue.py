#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'libjob_queue.py' is part of ERT - Ensemble based Reservoir Tool. 
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


import ctypes
import ert.util.libutil
import ert.util.SDP  as SDP
import ert.util.clib as clib


if SDP.RH_version() >= 4:
    LSF_HOME    = "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64"
else:
    LSF_HOME    = "/prog/LSF/7.0/linux2.4-glibc2.2-x86_64"

clib.load("libnsl.so" , "libnsl.so.1")
clib.load("libnsl.so.1")
clib.load("%s/lib/liblsf.so" % LSF_HOME)
clib.load("%s/lib/libbat.so" % LSF_HOME)

clib.load("libconfig.so" )
lib  = clib.load("libjob_queue.so")


