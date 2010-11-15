import ctypes
from   ert.cwrap.cwrap       import *
from   ert.util.tvector      import IntVector
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

    def union_with( self, other):
        return self.__ior__( other )

    def intersect_with( self, other):
        return self.__iand__( other )
    
    def from_param(self):
        return self.c_ptr

    def copy( self ):
        return self.__deep_copy__( {} )

    def reset(self):
        cfunc.reset( self )

    ##################################################################
    # Select functions

    def select_more( self , ecl_kw , limit):
        cfunc.select_more( self , ecl_kw , limit )

    def deselect_more( self , ecl_kw , limit):
        cfunc.deselect_more( self , ecl_kw , limit )

    def select_less( self , ecl_kw , limit):
        cfunc.select_less( self , ecl_kw , limit )

    def deselect_less( self , ecl_kw , limit):
        cfunc.deselect_less( self , ecl_kw , limit )

    def select_equal( self , ecl_kw , value ):
        cfunc.select_equal( self , ecl_kw , value )

    def deselect_equal( self , ecl_kw , value ):
        cfunc.deselect_equal( self , ecl_kw , value )

    def select_in_range( self , ecl_kw , lower_limit , upper_limit):
        cfunc.select_in_interval( self , ecl_kw , lower_limit , upper_limit)

    def deselect_in_range( self , ecl_kw , lower_limit , upper_limit):
        cfunc.deselect_in_interval( self , ecl_kw , lower_limit , upper_limit)

    def select_cmp_less( self , kw1 , kw2):
        cfunc.select_cmp_less( self , kw1 , kw2 )

    def deselect_cmp_less( self , kw1 , kw2):
        cfunc.deselect_cmp_less( self , kw1 , kw2 )

    def select_cmp_more( self , kw1 , kw2):
        cfunc.select_cmp_more( self , kw1 , kw2 )

    def deselect_cmp_more( self , kw1 , kw2):
        cfunc.deselect_cmp_more( self , kw1 , kw2 )

    def select_active( self ):
        cfunc.select_active( self )

    def deselect_active( self ):
        cfunc.deselect_active( self )

    def select_inactive( self ):
        cfunc.select_inactive( self )

    def deselect_inactive( self ):
        cfunc.deselect_inactive( self )

    def select_all( self ):
        cfunc.select_all( self )

    def deselect_all( self ):
        cfunc.deselect_all( self )

    def select_deep( self, depth):
        cfunc.select_deep_cells(self , depth)

    def deselect_deep( self, depth):
        cfunc.deselect_deep_cells(self , depth)

    def select_shallow( self, depth):
        cfunc.select_shallow_cells(self , depth)

    def deselect_shallow( self, depth):
        cfunc.deselect_shallow_cells(self , depth)
        
    def select_small( self , size_limit ):
        cfunc.select_small( self , size_limit )

    def deselect_small( self , size_limit ):
        cfunc.deselect_small( self , size_limit )

    def select_large( self , size_limit ):
        cfunc.select_large( self , size_limit )

    def deselect_large( self , size_limit ):
        cfunc.deselect_large( self , size_limit )

    def select_thin( self , size_limit ):
        cfunc.select_thin( self , size_limit )

    def deselect_thin( self , size_limit ):
        cfunc.deselect_thin( self , size_limit )

    def select_thick( self , size_limit ):
        cfunc.select_thick( self , size_limit )

    def deselect_thick( self , size_limit ):
        cfunc.deselect_thick( self , size_limit )

    def select_box( self , ijk1 , ijk2 ):
        cfunc.select_box( self , ijk1[0] , ijk2[0] , ijk1[1] , ijk2[1] , ijk1[2] , ijk2[2])

    def deselect_box( self , ijk1 , ijk2 ):
        cfunc.deselect_box( self , ijk1[0] , ijk2[0] , ijk1[1] , ijk2[1] , ijk1[2] , ijk2[2])

    def select_islice( self , i1 , i2):
        cfunc.select_islice( self , i1,i2)

    def deselect_islice( self , i1 , i2):
        cfunc.deselect_islice( self , i1,i2)

    def select_jslice( self , j1 , j2):
        cfunc.select_islice( self , j1,j2)

    def deselect_jslice( self , j1 , j2):
        cfunc.deselect_islice( self , j1,j2)

    def select_kslice( self , k1 , k2):
        cfunc.select_islice( self , k1,k2)

    def deselect_kslice( self , k1 , k2):
        cfunc.deselect_islice( self , k1,k2)

    def invert( self ):
        cfunc.invert_selection( self )
        
    ##################################################################
    # Functions to manipulate KW:
    
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
        c_ptr = cfunc.get_active_list( self )
        active_list = IntVector.ref( c_ptr , self )
        return active_list

    @property
    def global_list(self):
        c_ptr = cfunc.get_global_list( self )
        global_list = IntVector.ref( c_ptr , self )    
        return global_list

    @property
    def active_size( self ):
        return self.active_list.size

    @property
    def global_size( self ):
        return self.global_list.size    
    
    def kw_index_list(self , ecl_kw , force_active):
        c_ptr = cfunc.get_kw_index_list( self , ecl_kw , force_active)
        index_list = IntVector.ref( c_ptr , self )
        return index_list


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

cfunc.iadd_kw                    = cwrapper.prototype("void  ecl_region_kw_iadd( ecl_region , ecl_kw , ecl_kw , bool)")
cfunc.isub_kw                    = cwrapper.prototype("void  ecl_region_kw_isub( ecl_region , ecl_kw , ecl_kw , bool)")
cfunc.copy_kw                    = cwrapper.prototype("void  ecl_region_kw_copy( ecl_region , ecl_kw , ecl_kw , bool)")

cfunc.alloc_copy                 = cwrapper.prototype("long ecl_region_alloc_copy( ecl_region )")
cfunc.intersect                  = cwrapper.prototype("void ecl_region_intersection( ecl_region , ecl_region )")
cfunc.combine                    = cwrapper.prototype("void ecl_region_union( ecl_region , ecl_region )")

cfunc.get_kw_index_list          = cwrapper.prototype("long ecl_region_get_kw_index_list( ecl_region , ecl_kw , bool )")
cfunc.get_active_list            = cwrapper.prototype("long ecl_region_get_active_list( ecl_region )")
cfunc.get_global_list            = cwrapper.prototype("long ecl_region_get_global_list( ecl_region )")
cfunc.get_active_global          = cwrapper.prototype("long ecl_region_get_global_active_list( ecl_region )")

cfunc.select_cmp_less            = cwrapper.prototype("void ecl_region_cmp_select_less( ecl_region , ecl_kw , ecl_kw)")
cfunc.select_cmp_more            = cwrapper.prototype("void ecl_region_cmp_select_more( ecl_region , ecl_kw , ecl_kw)")
cfunc.deselect_cmp_less          = cwrapper.prototype("void ecl_region_cmp_deselect_less( ecl_region , ecl_kw , ecl_kw)")
cfunc.deselect_cmp_more          = cwrapper.prototype("void ecl_region_cmp_deselect_more( ecl_region , ecl_kw , ecl_kw)")

cfunc.select_islice              = cwrapper.prototype("void ecl_region_select_i1i2( ecl_region , int , int )")
cfunc.deselect_islice            = cwrapper.prototype("void ecl_region_deselect_i1i2( ecl_region , int , int )")
cfunc.select_jslice              = cwrapper.prototype("void ecl_region_select_j1j2( ecl_region , int , int )")
cfunc.deselect_jslice            = cwrapper.prototype("void ecl_region_deselect_j1j2( ecl_region , int , int )")
cfunc.select_kslice              = cwrapper.prototype("void ecl_region_select_k1k2( ecl_region , int , int )")
cfunc.deselect_kslice            = cwrapper.prototype("void ecl_region_deselect_k1k2( ecl_region , int , int )")

cfunc.select_deep_cells          = cwrapper.prototype("void ecl_region_select_deep_cells( ecl_region , double )")
cfunc.deselect_deep_cells        = cwrapper.prototype("void ecl_region_select_deep_cells( ecl_region , double )")
cfunc.select_shallow_cells       = cwrapper.prototype("void ecl_region_select_shallow_cells( ecl_region , double )")
cfunc.deselect_shallow_cells     = cwrapper.prototype("void ecl_region_select_shallow_cells( ecl_region , double )")

cfunc.select_small               = cwrapper.prototype("void ecl_region_select_small_cells( ecl_region , double )")
cfunc.deselect_small             = cwrapper.prototype("void ecl_region_deselect_small_cells( ecl_region , double )")
cfunc.select_large               = cwrapper.prototype("void ecl_region_select_large_cells( ecl_region , double )")
cfunc.deselect_large             = cwrapper.prototype("void ecl_region_deselect_large_cells( ecl_region , double )")

cfunc.select_thin                = cwrapper.prototype("void ecl_region_select_thin_cells( ecl_region , double )")
cfunc.deselect_thin              = cwrapper.prototype("void ecl_region_deselect_thin_cells( ecl_region , double )")
cfunc.select_thick               = cwrapper.prototype("void ecl_region_select_thick_cells( ecl_region , double )")
cfunc.deselect_thick             = cwrapper.prototype("void ecl_region_deselect_thick_cells( ecl_region , double )")

cfunc.select_active              = cwrapper.prototype("void ecl_region_select_active_cells( ecl_region )")
cfunc.select_inactive            = cwrapper.prototype("void ecl_region_select_inactive_cells( ecl_region )")
cfunc.deselect_active            = cwrapper.prototype("void ecl_region_deselect_active_cells( ecl_region )")
cfunc.deselect_inactive          = cwrapper.prototype("void ecl_region_deselect_inactive_cells( ecl_region )")

