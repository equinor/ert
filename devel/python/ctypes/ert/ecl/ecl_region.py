import ctypes
from   ert.cwrap.cwrap       import *
from   ecl_kw                import ECL_INT_TYPE , ECL_REAL_TYPE , ECL_DOUBLE_TYPE
import libecl


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


    def set_kw( self , ecl_kw , value):
        type = ecl_kw.type
        if type == ECL_INT_TYPE:
            cfunc.set_kw_int( ecl_kw , value )
        elif type == ECL_FLOAT_TYPE:
            cfunc.set_kw_float( ecl_kw , value )
        elif type == ECL_DOUBLE_TYPE:
            cfunc.set_kw_double( ecl_kw , value )
        raise Exception("set_kw() only supported for INT/FLOAT/DOUBLE")



# 2. Creating a wrapper object around the libecl library.
cwrapper = CWrapper( libecl.lib )
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

cfunc.set_kw_int                 = cwrapper.prototype("void ecl_region_set_kw_int( ecl_region , ecl_kw , int) ")
cfunc.set_kw_float               = cwrapper.prototype("void ecl_region_set_kw_float( ecl_region , ecl_kw , float) ")
cfunc.set_kw_double              = cwrapper.prototype("void ecl_region_set_kw_double( ecl_region , ecl_kw , double) ")

