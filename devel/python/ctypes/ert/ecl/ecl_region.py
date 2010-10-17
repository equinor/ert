import ctypes
from   ert.cwrap.cwrap       import *
from   ecl_kw                import ECL_INT_TYPE , ECL_FLOAT_TYPE , ECL_DOUBLE_TYPE
import libecl


class EclRegion:
    def __init__(self , grid , preselect , c_ptr = None):
        self.grid  = grid
        if c_ptr:
            self.c_ptr = c_ptr
        else:
            self.c_ptr = cfunc.alloc( grid , preselect )
            

    def __deep_copy__(self , memo):
        return EclRegion( self.grid , False , c_ptr = cfunc.alloc_copy( self ))

    def __iand__(self , other):
        if hasattr( other , "ecl_region_instance"):
            cfunc.intersect( self, other)
        else:
            raise TypeError("Ecl region can only intersect with other EclRegion instances")
            
        return self

    def __isub__(self , other):
        return self.__iand__( other )


    def __ior__(self , other):
        if hasattr( other , "ecl_region_instance"):
            cfunc.combine( self, other)
        else:
            raise TypeError("Ecl region can only be combined with other EclRegion instances")
            
        return self

    
    def __iadd__(self , other):
        return self.__ior__( other )


    def __or__(self , other):
        new_region = self.copy()
        new_region.__iadd__( other )
        return new_region

    def __and__(self , other):
        new_region = self.copy()
        new_region.__isub__( other )
        return new_region

    def __add__(self , other):
        return self.__add__( other )

    def __sub__( self, other):
        return self.__sub__( other )

    def __del__( self ):
        cfunc.free( self )

    def from_param(self):
        return self.c_ptr

    def copy( self ):
        return self.__deep_copy__( {} )

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
        
    def select_box( self , ijk1 , ijk2 ):
        cfunc.select_box( self , ijk1[0] , ijk2[0] , ijk1[1] , ijk2[1] , ijk1[2] , ijk2[2])

    def invert( self ):
        cfunc.invert_selection( self )

    
    def iadd_kw( self , target_kw , delta_kw , force_active = False):
        if target_kw.assert_binary( delta_kw ):
            cfunc.iadd_kw( self , target_kw , delta_kw , force_active )
        else:
            raise TypeError("Type mismatch")

    def copy_kw( self , target_kw , src_kw , force_active = False):
        if target_kw.assert_binary( src_kw ):
            cfunc.copy_kw( self , target_kw , src_kw , force_active )
        else:
            raise TypeError("Type mismatch")
        
    def set_kw( self , ecl_kw , value , force_active = False):
        type = ecl_kw.__type__
        if type == ECL_INT_TYPE:
            cfunc.set_kw_int( self , ecl_kw , value , force_active)
        elif type == ECL_FLOAT_TYPE:
            cfunc.set_kw_float( self , ecl_kw , value , force_active)
        elif type == ECL_DOUBLE_TYPE:
            cfunc.set_kw_double( self , ecl_kw , value , force_active )
        else:
            raise Exception("set_kw() only supported for INT/FLOAT/DOUBLE")

    def shift_kw( self , ecl_kw , shift , force_active = False):
        type = ecl_kw.__type__
        if type == ECL_INT_TYPE:
            cfunc.shift_kw_int( self , ecl_kw , shift , force_active)
        elif type == ECL_FLOAT_TYPE:
            cfunc.shift_kw_float( self , ecl_kw , shift , force_active)
        elif type == ECL_DOUBLE_TYPE:
            cfunc.shift_kw_double( self , ecl_kw , shift , force_active )
        else:
            raise Exception("shift_kw() only supported for INT/FLOAT/DOUBLE")

    def scale_kw( self , ecl_kw , scale , force_active = False):
        type = ecl_kw.__type__
        if type == ECL_INT_TYPE:
            cfunc.scale_kw_int( self , ecl_kw , scale , force_active)
        elif type == ECL_FLOAT_TYPE:
            cfunc.scale_kw_float( self , ecl_kw , scale , force_active)
        elif type == ECL_DOUBLE_TYPE:
            cfunc.scale_kw_double( self , ecl_kw , scale , force_active )
        else:
            raise Exception("scale_kw() only supported for INT/FLOAT/DOUBLE")

    def ecl_region_instance( self ):
        return True


    @property
    def active_list(self):
        a             = cfunc.get_active_list( self )
        a.size        = cfunc.get_active_size( self )
        a.__parent__  = self  # Inhibit GC
        return a

    @property
    def global_list(self):
        a             = cfunc.get_global_list( self )
        a.size        = cfunc.get_global_size( self )
        a.__parent__  = self  # Inhibit GC
        return a

    @property
    def active_size( self ):
        return cfunc.active_size( self )

    @property
    def global_size( self ):
        return cfunc.global_size( self )

    


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

cfunc.select_in_interval         = cwrapper.prototype("void ecl_region_select_in_interval( ecl_region, ecl_kw , float , float )")
cfunc.deselect_in_interval       = cwrapper.prototype("void ecl_region_deselect_in_interval( ecl_region, ecl_kw, float , float )")

cfunc.invert_selection           = cwrapper.prototype("void ecl_region_invert_selection( ecl_region )")

cfunc.active_size                = cwrapper.prototype("int  ecl_region_get_active_size( ecl_region )")
cfunc.global_size                = cwrapper.prototype("int  ecl_region_get_global_size( ecl_region )")
cfunc.active_set                 = cwrapper.prototype("int* ecl_region_get_active_list( ecl_region )")
cfunc.global_set                 = cwrapper.prototype("int* ecl_region_get_global_list( ecl_region )")

cfunc.set_kw_int                 = cwrapper.prototype("void ecl_region_set_kw_int( ecl_region , ecl_kw , int, bool) ")
cfunc.set_kw_float               = cwrapper.prototype("void ecl_region_set_kw_float( ecl_region , ecl_kw , float, bool ) ")
cfunc.set_kw_double              = cwrapper.prototype("void ecl_region_set_kw_double( ecl_region , ecl_kw , double , bool) ")

cfunc.shift_kw_int                 = cwrapper.prototype("void ecl_region_shift_kw_int( ecl_region , ecl_kw , int, bool) ")
cfunc.shift_kw_float               = cwrapper.prototype("void ecl_region_shift_kw_float( ecl_region , ecl_kw , float, bool ) ")
cfunc.shift_kw_double              = cwrapper.prototype("void ecl_region_shift_kw_double( ecl_region , ecl_kw , double , bool) ")

cfunc.scale_kw_int                 = cwrapper.prototype("void ecl_region_scale_kw_int( ecl_region , ecl_kw , int, bool) ")
cfunc.scale_kw_float               = cwrapper.prototype("void ecl_region_scale_kw_float( ecl_region , ecl_kw , float, bool ) ")
cfunc.scale_kw_double              = cwrapper.prototype("void ecl_region_scale_kw_double( ecl_region , ecl_kw , double , bool) ")

cfunc.select_box                 = cwrapper.prototype("void ecl_region_select_from_ijkbox(ecl_region , int , int , int , int , int , int)")     

cfunc.get_active_list            = cwrapper.prototype("int* ecl_region_get_active_list( ecl_region )")
cfunc.get_global_list            = cwrapper.prototype("int* ecl_region_get_global_list( ecl_region )")
cfunc.get_active_size            = cwrapper.prototype("int   ecl_region_get_active_size( ecl_region )")
cfunc.get_global_size            = cwrapper.prototype("int   ecl_region_get_global_size( ecl_region )")
cfunc.iadd_kw                    = cwrapper.prototype("void  ecl_region_kw_iadd( ecl_region , ecl_kw , ecl_kw , bool)")
cfunc.copy_kw                    = cwrapper.prototype("void  ecl_region_kw_copy( ecl_region , ecl_kw , ecl_kw , bool)")

cfunc.alloc_copy                 = cwrapper.prototype("long ecl_region_alloc_copy( ecl_region )")
cfunc.intersect                  = cwrapper.prototype("void ecl_region_select_intersection( ecl_region , ecl_region )")
cfunc.combine                    = cwrapper.prototype("void ecl_region_select_union( ecl_region , ecl_region )")
