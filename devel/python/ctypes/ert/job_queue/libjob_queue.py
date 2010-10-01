import ctypes
import ert.util.SDP as SDP

if SDP.RH_version() >= 4:
    LSF_HOME    = "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64"
else:
    LSF_HOME    = "/prog/LSF/7.0/linux2.4-glibc2.2-x86_64"


ctypes.CDLL("libnsl.so"                     , ctypes.RTLD_GLOBAL)
ctypes.CDLL("%s/lib/liblsf.so" % LSF_HOME       , ctypes.RTLD_GLOBAL)
ctypes.CDLL("%s/lib/libbat.so" % LSF_HOME       , ctypes.RTLD_GLOBAL)

ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libutil.so"   , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libconfig.so" , ctypes.RTLD_GLOBAL)
lib  = ctypes.CDLL("libjob_queue.so"  , ctypes.RTLD_GLOBAL)


