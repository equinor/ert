import sys
from   ert.cwrap.cwrap       import *

class FortIO:
    def __init__(self , filename , mode , fmt_file = False , endian_flip = True):
        self.c_ptr = cfunc.fortio_fopen( filename , mode , endian_flip , fmt_file)
        self.file_open = True 

    def from_param(self):
        return self.c_ptr

    def __del__(self):
        if self.file_open:
            cfunc.fortio_close( self )
            
    def close( self ):
        cfunc.fortio_close( self )
        self.file_open = False



# 1. Loading the necessary C-libraries.
ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libutil.so" , ctypes.RTLD_GLOBAL)
libecl  = ctypes.CDLL("libecl.so"  , ctypes.RTLD_GLOBAL)


# 2. Creating a wrapper object around the libecl library, 
#    registering the type map : ecl_kw <-> EclKW
cwrapper = CWrapper( libecl )
cwrapper.registerType("fortio" , FortIO )


# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("fortio")
cfunc.fortio_fopen = cwrapper.prototype("long fortio_fopen(char* , char* , bool)")
cfunc.fortio_close = cwrapper.prototype("void fortio_fclose( fortio )")


