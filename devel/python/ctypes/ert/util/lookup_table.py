import sys
import ctypes
import libutil
from   ert.cwrap.cwrap       import *



class LookupTable(object):
    
    def __init__(self):
        self.c_ptr = cfunc.alloc()
        
    def from_param(self):
        return self.c_ptr
        
    @property 
    def max(self):
        return cfunc.max( self )

    @property 
    def min(self):
        return cfunc.min( self )

    @property 
    def arg_max(self):
        return cfunc.arg_max( self )

    @property 
    def arg_min(self):
        return cfunc.arg_min( self )
    
    def interp( self , x ):
        return cfunc.interp(self , x )

    def append( self , x , y ):
        cfunc.append( self , x , y )
        
    @property
    def size(self):
        return cfunc.size( self )


    def __del__(self):
        cfunc.free( self )

        
CWrapper.registerType( "lookup_table" , LookupTable )
cwrapper = CWrapper( libutil.lib )
cfunc    = CWrapperNameSpace("lookup_table")

cfunc.alloc     = cwrapper.prototype("c_void_p   lookup_table_alloc_empty()")
cfunc.max       = cwrapper.prototype("double     lookup_table_get_max_value( lookup_table )") 
cfunc.min       = cwrapper.prototype("double     lookup_table_get_min_value( lookup_table )") 
cfunc.arg_max   = cwrapper.prototype("double     lookup_table_get_max_arg( lookup_table )") 
cfunc.arg_min   = cwrapper.prototype("double     lookup_table_get_min_arg( lookup_table )") 
cfunc.append    = cwrapper.prototype("void      lookup_table_append( lookup_table , double , double )") 
cfunc.size      = cwrapper.prototype("int       lookup_table_get_size( lookup_table )")
cfunc.interp      = cwrapper.prototype("double     lookup_table_interp( lookup_table , double)")
cfunc.free        = cwrapper.prototype("void lookup_table_free( lookup_table )")
