import ctypes

ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libutil.so" , ctypes.RTLD_GLOBAL)
lib  = ctypes.CDLL("libecl.so"  , ctypes.RTLD_GLOBAL)
