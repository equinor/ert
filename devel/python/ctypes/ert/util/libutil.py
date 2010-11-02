import ctypes
import ert.util.clib as clib

clib.load("libz" , "libz.so.1")
#clib.load("/usr/lib/libg2c.so.0" , "/usr/lib/libg2c.so.0.0.0")
clib.load("libblas.so")
clib.load("liblapack.so")
lib = clib.load("libutil.so")
