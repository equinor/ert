import ctypes
from   ert.cwrap.cwrap       import *
import ecl_grid



class EclRegion:
    def __init__(self , grid , preselect):
        self.grid  = grid
        self.c_ptr = cfunc.alloc( grid , preselect )
        
    def __del__( self ):
        cfunc.free( self )

    def from_param(self):
        return self.c_ptr

    def reset(self):
        cfunc.reset( self )

    def select_more( self , ecl_kw , limit):
        cfunc.select_more( self , ecl_kw , limit )

    def select_less( self , ecl_kw , limit):
        cfunc.select_less( self , ecl_kw , limit )

    def select_equal( self , ecl_kw , value ):
        cfunc.select_equal( self , ecl_kw , value )

    def select_all( self ):
        cfunc.select_all( self )

    def invert( self ):
        cfunc.invert_selection( self )

    def active_size( self ):
        return cfunc.active_size( self )

    def global_size( self ):
        return cfunc.global_size( self )
    
    
    def active_set( self ):
        list = cfunc.active_set( self )
        list.size = cfunc.active_size( self )
        return list

    
    def global_set( self ):
        list = cfunc.global_set( self )
        list.size = cfunc.global_size( self )
        return list




# 1. Loading the necessary C-libraries.
ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libutil.so" , ctypes.RTLD_GLOBAL)
libecl  = ctypes.CDLL("libecl.so"  , ctypes.RTLD_GLOBAL)

# 2. Creating a wrapper object around the libecl library.
cwrapper = CWrapper( libecl )
cwrapper.registerType( "ecl_region" , EclRegion )

# 3. Installing the c-functions used to manipulate.
cfunc = CWrapperNameSpace("ecl_region")

cfunc.alloc                      = cwrapper.prototype("long ecl_region_alloc( ecl_grid , bool )")
cfunc.free                       = cwrapper.prototype("void ecl_region_free( ecl_region )")     
cfunc.reset                      = cwrapper.prototype("void ecl_region_reset( ecl_region )")

cfunc.select_all                 = cwrapper.prototype("void ecl_region_select_all( ecl_region )")
cfunc.deselect_all               = cwrapper.prototype("void ecl_region_deselect_all( ecl_region )")

cfunc.select_equal               = cwrapper.prototype("void ecl_region_select_all( ecl_region , ecl_kw , int )")
cfunc.deselect_equal             = cwrapper.prototype("void ecl_region_deselect_all( ecl_region , ecl_kw , int)")

cfunc.select_less                = cwrapper.prototype("void ecl_region_select_smaller( ecl_region , ecl_kw , float )")
cfunc.deselect_less              = cwrapper.prototype("void ecl_region_deselect_smaller( ecl_region , ecl_kw , float )")

cfunc.select_more                = cwrapper.prototype("void ecl_region_select_larger( ecl_region , ecl_kw , float )")
cfunc.deselect_more              = cwrapper.prototype("void ecl_region_deselect_larger( ecl_region , ecl_kw , float )")

cfunc.invert_selection           = cwrapper.prototype("void ecl_region_invert_selection( ecl_region )")

cfunc.active_size                = cwrapper.prototype("int  ecl_region_get_active_size( ecl_region )")
cfunc.global_size                = cwrapper.prototype("int  ecl_region_get_global_size( ecl_region )")
cfunc.active_set                 = cwrapper.prototype("int* ecl_region_get_active_list( ecl_region )")
cfunc.global_set                 = cwrapper.prototype("int* ecl_region_get_global_list( ecl_region )")

