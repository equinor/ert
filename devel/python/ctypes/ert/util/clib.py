import ctypes


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
    
    raise ImportError("Sorry - failed to load shared library:%s." % lib)
