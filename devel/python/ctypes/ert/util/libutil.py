import ctypes
import ert.util.clib as clib

clib.load("libz" , "libz.so.1")
clib.load("libg2c")
clib.load("libblas.so" , "/usr/lib64/libblas.so.3.0.3")
clib.load("liblapack.so")
lib = clib.load("libutil.so")
