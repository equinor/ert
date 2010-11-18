import ctypes
import ert.util.clib as clib
import ert.util.SDP  as SDP

clib.load("libz" , "libz.so.1")

if SDP.RH_version() < 4:
    clib.load("libg2c.so.0")

clib.load("libblas.so" , "libblas.so.3")
clib.load("liblapack.so")
lib = clib.load("libutil.so")
