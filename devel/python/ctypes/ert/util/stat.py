import ctypes
import libutil
import tvector 
from   ert.cwrap.cwrap       import *

def quantile( data , q ):
    return cfunc.quantile( data , q )


def quantile_sorted( data , q ):
    return cfunc.quantile_sorted( data , q )


cwrapper = CWrapper( libutil.lib )
cfunc    = CWrapperNameSpace("stat")


cfunc.quantile        = cwrapper.prototype("double statistics_empirical_quantile( double_vector , double )")
cfunc.quantile_sorted = cwrapper.prototype("double statistics_empirical_quantile( double_vector , double )")
