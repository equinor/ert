import ctypes
from   ert.cwrap.cwrap       import *
import fortio



# Enum defintion from ecl_util.h
ECL_CHAR_TYPE   = 0
ECL_REAL_TYPE   = 1
ECL_DOUBLE_TYPE = 2
ECL_INT_TYPE    = 3
ECL_BOOL_TYPE   = 4
ECL_MESS_TYPE   = 5


class EclKW:
    def __init__(self , parent , c_ptr):
        self.__parent   = parent   # Hold on to the parent to inhibit GC
        self.c_ptr      = c_ptr
        self.data_owner = False
    
    def from_param(self):
        return self.c_ptr

    def __len__( self ):
        return cfunc.get_size( self )
    
    def __del__(self):
        if self.data_owner:
            print "Calling ecl_kw_free()"
            cfunc.free( self )

    def __deep_copy__(self , memo):
        c_ptr  = cfunc.copyc( self )
        ecl_kw = EclKW( None , c_ptr )
        ecl_kw.data_owner = True
        return ecl_kw

        
    def deep_copy( self ):
        ecl_kw = self.__deep_copy__( {} )
        return ecl_kw

    @property
    def size(self):
        return cfunc.get_size( self )

    @property
    def type( self ):
        __type = cfunc.get_type( self )
        # enum ecl_type_enum from ecl_util.h
        if __type == ECL_CHAR_TYPE:
            return "CHAR"
        if __type == ECL_REAL_TYPE:
            return "REAL"
        if __type == ECL_DOUBLE_TYPE:
            return "DOUB"
        if __type == ECL_INT_TYPE:
            return "INTE"
        if __type == ECL_BOOL_TYPE:
            return "BOOL"
        if __type == ECL_MESS_TYPE:
            return "MESS"

        
    def iget( self , index ):
        __type = cfunc.get_type( self )
        if __type == ECL_CHAR_TYPE:
            value = cfunc.iget_char_ptr( self , index )

        if __type == ECL_REAL_TYPE:
            value = cfunc.iget_float( self , index )

        if __type == ECL_REAL_TYPE:
            value = cfunc.iget_double( self , index )

        if __type == ECL_INT_TYPE:
            value = cfunc.iget_int( self , index )

        if __type == ECL_BOOL_TYPE:
            value = cfunc.iget_bool( self , index )
            
        return value
    
    @property
    def array(self):
        type = cfunc.get_type( self )
        if type == ECL_INT_TYPE:
            a = cfunc.int_ptr( self )
        elif type == ECL_REAL_TYPE:
            a = cfunc.float_ptr( self )
        elif type == ECL_DOUBLE_TYPE:
            a = cfunc.double_ptr( self )
        else:
            a = None

        if not a == None:
            a.size       = cfunc.get_size( self )
            a.__parent__ = self  # Inhibit GC
        return a
        
    def numpy_array(self):
        pass


    def fwrite( self , fortio ):
        cfunc.fwrite( self , fortio )



#################################################################

# 1. Loading the necessary C-libraries.
ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libutil.so" , ctypes.RTLD_GLOBAL)
libecl  = ctypes.CDLL("libecl.so"  , ctypes.RTLD_GLOBAL)


# 2. Creating a wrapper object around the libecl library, 
#    registering the type map : ecl_kw <-> EclKW
cwrapper = CWrapper( libecl )
cwrapper.registerType( "ecl_kw" , EclKW )

# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_kw")
cfunc.get_size                   = cwrapper.prototype("int ecl_kw_get_size( ecl_kw )")
cfunc.get_type                   = cwrapper.prototype("int ecl_kw_get_type( ecl_kw )")
cfunc.iget_char_ptr              = cwrapper.prototype("char* ecl_kw_iget_char_ptr( ecl_kw , int )")
cfunc.iget_bool                  = cwrapper.prototype("bool ecl_kw_iget_bool( ecl_kw , int)")
cfunc.iget_int                   = cwrapper.prototype("int ecl_kw_iget_int( ecl_kw , int )")
cfunc.iget_double                = cwrapper.prototype("double ecl_kw_iget_double( ecl_kw , int )")
cfunc.iget_float                 = cwrapper.prototype("float ecl_kw_iget_float( ecl_kw , int)")
cfunc.float_ptr                  = cwrapper.prototype("float* ecl_kw_get_float_ptr( ecl_kw )")
cfunc.free                       = cwrapper.prototype("void ecl_kw_free( ecl_kw )")
cfunc.copyc                      = cwrapper.prototype("long ecl_kw_alloc_copy( ecl_kw )")
cfunc.fwrite                     = cwrapper.prototype("void ecl_kw_fwrite( ecl_kw , fortio )")
