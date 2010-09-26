import time
import datetime
import ctypes
from   ert.cwrap.cwrap       import *


class CFILE:
    def __init__(self , py_file):
        self.c_ptr = cfunc.fdopen( py_file.fileno() , 'w' )
        self.py_file = py_file

    def from_param( self ):
        return self.c_ptr

    def flush( self ):
        cfunc.flush( self )
    
    def __del__(self):
        print "Calling CFILe __del__"
        pass




libc  = ctypes.CDLL("/lib64/tls/libc.so.6" , ctypes.RTLD_GLOBAL )
cwrapper = CWrapper( libc )
cwrapper.registerType("FILE" , CFILE )
cfunc = CWrapperNameSpace("libc")
cfunc.fdopen = cwrapper.prototype("long fdopen( int , char* )")
cfunc.flush  = cwrapper.prototype("int  fflush( FILE )")
