import sys
import libecl
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
        return self.c_ptr

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
cfunc.fortio_fopen = cwrapper.prototype("long fortio_fopen(char* , char* , bool)")
cfunc.fortio_close = cwrapper.prototype("void fortio_fclose( fortio )")


