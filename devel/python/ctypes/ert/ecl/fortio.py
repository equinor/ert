#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'fortio.py' is part of ERT - Ensemble based Reservoir Tool. 
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
import libecl
import ctypes
from   ert.cwrap.cwrap       import *


# The FortIO class is a thin wrapper around the C struct fortio. The
# FortIO class is created to facilitate reading and writing binary
# fortran files. The python implementation (currently) only supports
# instantiation and subsequent use in a C function, but it would be
# simple to wrap the low level read/write functions for Python access as
# well.

class FortIO:
    def __init__(self , filename , mode , fmt_file = False , endian_flip = True):
        self.c_ptr = cfunc.fortio_fopen( filename , mode , endian_flip , fmt_file)
        self.file_open = True 

    def from_param(self):
        return ctypes.c_void_p( self.c_ptr )

    # Implements normal Python semantics - close on delete.
    def __del__(self):
        if self.file_open:
            cfunc.fortio_close( self )
            
    def close( self ):
        cfunc.fortio_close( self )
        self.file_open = False



# 2. Creating a wrapper object around the libecl library, 
#    registering the type map : fortio <-> FortIO
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType("fortio" , FortIO )


# 3. Installing the c-functions used to manipulate fortio instances.
#    These functions are used when implementing the FortIO class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("fortio")
cfunc.fortio_fopen = cwrapper.prototype("c_void_p fortio_fopen(char* , char* , bool)")
cfunc.fortio_close = cwrapper.prototype("void     fortio_fclose( fortio )")


