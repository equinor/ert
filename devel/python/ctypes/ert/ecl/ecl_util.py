from    ert.cwrap.cwrap       import *
import  libecl



def get_num_cpu( datafile ):
    return cfunc.get_num_cpu( datafile )




cwrapper = CWrapper( libecl.lib )
cfunc    = CWrapperNameSpace("ecl_util")
cfunc.get_num_cpu    = cwrapper.prototype("int ecl_util_get_num_cpu( char* )")
