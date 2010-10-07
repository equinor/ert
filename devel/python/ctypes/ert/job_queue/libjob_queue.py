import ctypes
import ert.util.libutil
import ert.util.SDP  as SDP
import ert.util.clib as clib

if SDP.RH_version() >= 4:
    LSF_HOME    = "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64"
else:
    LSF_HOME    = "/prog/LSF/7.0/linux2.4-glibc2.2-x86_64"


clib.load("libnsl.so" , "libnsl.so.1")
clib.load("libnsl.so.1")
clib.load("%s/lib/liblsf.so" % LSF_HOME)
clib.load("%s/lib/libbat.so" % LSF_HOME)

clib.load("libconfig.so" )
lib  = clib.load("libjob_queue.so")


