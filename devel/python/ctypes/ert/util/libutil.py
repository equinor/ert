import ctypes
import ert.util.clib as clib

clib.load("libz" , "libz.so.1")
clib.load("libblas.so")
clib.load("liblapack.so")
lib = clib.load("libutil.so")
