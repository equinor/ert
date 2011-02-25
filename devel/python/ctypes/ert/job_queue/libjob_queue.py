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

import os
import ctypes
import ert.util.libutil
import ert.util.SDP  as SDP
import ert.util.clib as clib


# We try to import the lsf libraries unconditionally. If the import
# fails we assume the lsf_driver has been built without LSF
# support. If the lsf_driver has indeed been built with LSF support,
# and we now fail to load the lsf libraries it will crash and burn
# when we try to load the job_queue library ten lines further down.
#
# We use the environment variable LSF_LIBDIR to determine where the
# LSF libraries are located, if that variable is not set we just try a
# wild shot and see what the dynamic linker can come up with - that
# probably fails.


LSF_LIBDIR = os.getenv("LSF_LIBDIR")
try:
    clib.load("libnsl.so" , "libnsl.so.1")
    clib.load("libnsl.so.1")
    if LSF_LIBDIR:
        clib.load("%s/liblsf.so" % LSF_LIBDIR)
        clib.load("%s/libbat.so" % LSF_LIBDIR)
    else:
        clib.load( "liblsf.so" )
        clib.load( "libbat.so" )
    HAVE_LSF = True
except:
    HAVE_LSF = False


    
clib.load("libconfig.so" )
try:
    lib  = clib.load("libjob_queue.so")
except:
    if HAVE_LSF == False:
        sys.stderr.write("** Failed to load the libjob_queue library, \n")
        sys.stderr.write("** have previosuly failed to load the LSF\n")
        sys.stderr.write("** libraries liblsf & libbat - that might be\n")
        sys.stderr.wriet("** the reason ... ")
        if LSF_LIBDIR:
            sys.stderr.write("** LSF_LIBDIR = %s\n" % LSF_LIBDIR)
        else:
            sys.stderr.write("** LSF_LIBDIR = <NOT SET>\n")
    sys.exit("Failed to load library: libjob_queue")


