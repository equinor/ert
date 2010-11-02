import ctypes
import ert.util.clib as clib
from   ert.util.SDP  import RH_version

clib.load("libz" , "libz.so.1")

if RH_version() < 4:
    clib.load("libg2c.so.0")

clib.load("libblas.so")
clib.load("liblapack.so")
lib = clib.load("libutil.so")
