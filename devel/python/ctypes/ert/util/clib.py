import ctypes
import os

# The libraries can be given in alternative forms like this:
#
#    load("libz.so" , "libz.so.1" , "libz.so.1.2.1.2" , "libZ-fucker.so")
#
# Will return a handle to the first successfull load, and raise
# ImportError if none of the loads succeed.

def load( *lib_list ):
    dll = None
    
    for lib in lib_list:
        try:
            dll = ctypes.CDLL( lib , ctypes.RTLD_GLOBAL )
            return dll
        except:
            pass
    
    error_msg = "Sorry - failed to load shared library:%s\n\nTried in: " % lib_list[0]
    path_list = os.getenv("LD_LIBRARY_PATH").split(":")
    for path in path_list:
        error_msg += path + "\n          "

    raise ImportError( error_msg )
