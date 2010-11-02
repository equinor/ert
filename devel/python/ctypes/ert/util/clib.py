import ctypes
import os

# The shared libraries typically exist under several different names,
# with different level of version detail. Unfortunately the same
# library can exist under different names on the different Statoil
# computers, to support this the load function can get several
# arguments like:
#
#    load("libz.so" , "libz.so.1" , "libz.so.1.2.1.2" , "libZ-fucker.so")
#
# Will return a handle to the first successfull load, and raise
# ImportError if none of the loads succeed.

def load( *lib_list ):
    dll = None
    
    for lib in lib_list:
        print "Trying:%s" % lib
        try:
            dll = ctypes.CDLL( lib , ctypes.RTLD_GLOBAL )
            return dll
        except Exception, msg:
            print "msg:%s" % msg
            pass
    
    error_msg = "Sorry - failed to load shared library:%s\n\nTried in: " % lib_list[0]
    path_list = os.getenv("LD_LIBRARY_PATH").split(":")
    for path in path_list:
        error_msg += path + "\n          "

    raise ImportError( error_msg )
