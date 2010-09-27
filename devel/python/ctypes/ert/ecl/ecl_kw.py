from    ert.cwrap.cwrap       import *
import  numpy
import  fortio
import  libecl
from    ert.util.cfile   import CFILE



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
        self.data_ptr   = None
        self.__type = cfunc.get_type( self )
        if self.__type == ECL_INT_TYPE:
            self.data_ptr = cfunc.int_ptr( self )
            self.dtype    = int        
        elif self.__type == ECL_REAL_TYPE:
            self.data_ptr = cfunc.float_ptr( self )
            self.dtype    = float
        elif self.__type == ECL_DOUBLE_TYPE:
            self.data_ptr = cfunc.double_ptr( self )
            self.dtype    = double        
        else:
            # Iteration not supported ...
            self.data_ptr = None
            self.dtype = None
            

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

    def __getitem__(self, index ):
        if isinstance( index , types.IntType):
            length = self.__len__()
            if index < 0 or index >= length:
                raise IndexError
            else:
                if self.data_ptr:
                    return self.data_ptr[ index ]
                else:
                    if self.__type == ECL_BOOL_TYPE:
                        return cfunc.iget_bool( self, index)
                    elif self.__type == ECL_CHAR_TYPE:
                        return cfunc.iget_char_ptr( self , index )
                    else:
                        raise TypeError("Internal implementation error ...")
        else:
            raise TypeError("Index should be integer type")


    def __setitem__(self, index ,value):
        if isinstance( index , types.IntType):
            length = self.__len__()
            if index < 0 or index >= length:
                raise IndexError
            else:
                if self.data_ptr:
                    self.data_ptr[ index ] = value
                else:
                    if self.__type == ECL_BOOL_TYPE:
                        cfunc.iset_bool( self , index , value)
                    elif self.__type == ECL_CHAR_TYPE:
                        return cfunc.iset_char_ptr( self , index , value)
                    else:
                        raise SystemError("Internal implementation error ...")
        else:
            raise TypeError("Index should be integer type")

        
    def deep_copy( self ):
        ecl_kw = self.__deep_copy__( {} )
        return ecl_kw

    @property
    def size(self):
        return cfunc.get_size( self )
    
    @property
    def name( self ):
        return cfunc.get_header( self )

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

    @property
    def header( self ):
        return (self.name , self.size , self.type)


    def iget( self , index ):
        return self.__getitem__( index )


    @property
    def array(self):
        a = self.data_ptr
        if not a == None:
            a.size        = cfunc.get_size( self )
            a.__parent__  = self  # Inhibit GC
        return a


    @property
    def numpy_array( self ):
        if self.data_ptr:
            a = self.array
            value = numpy.zeros( a.size , dtype = self.dtype)
            for i in range( a.size ):
                value[i] = a[i]

    def fwrite( self , fortio ):
        cfunc.fwrite( self , fortio )

    def write_grdecl( self , file ):
        cfile = CFILE( file )
        cfunc.fprintf_grdecl( self , cfile )
        

#################################################################

# 1. Loading the necessary C-libraries.

# 2. Creating a wrapper object around the libecl library, 
#    registering the type map : ecl_kw <-> EclKW
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType( "ecl_kw" , EclKW )

# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_kw")
cfunc.get_size                   = cwrapper.prototype("int ecl_kw_get_size( ecl_kw )")
cfunc.get_type                   = cwrapper.prototype("int ecl_kw_get_type( ecl_kw )")
cfunc.iget_char_ptr              = cwrapper.prototype("char* ecl_kw_iget_char_ptr( ecl_kw , int )")
cfunc.iset_char_ptr              = cwrapper.prototype("void ecl_kw_iset_char_ptr( ecl_kw , int , char*)")
cfunc.iget_bool                  = cwrapper.prototype("bool ecl_kw_iget_bool( ecl_kw , int)")
cfunc.iset_bool                  = cwrapper.prototype("bool ecl_kw_iset_bool( ecl_kw , int, bool)")
cfunc.iget_int                   = cwrapper.prototype("int ecl_kw_iget_int( ecl_kw , int )")
cfunc.iget_double                = cwrapper.prototype("double ecl_kw_iget_double( ecl_kw , int )")
cfunc.iget_float                 = cwrapper.prototype("float ecl_kw_iget_float( ecl_kw , int)")
cfunc.float_ptr                  = cwrapper.prototype("float* ecl_kw_get_float_ptr( ecl_kw )")
cfunc.int_ptr                    = cwrapper.prototype("int* ecl_kw_get_int_ptr( ecl_kw )")
cfunc.double_ptr                 = cwrapper.prototype("double* ecl_kw_get_double_ptr( ecl_kw )")
cfunc.free                       = cwrapper.prototype("void ecl_kw_free( ecl_kw )")
cfunc.copyc                      = cwrapper.prototype("long ecl_kw_alloc_copy( ecl_kw )")
cfunc.fwrite                     = cwrapper.prototype("void ecl_kw_fwrite( ecl_kw , fortio )")
cfunc.get_header                 = cwrapper.prototype("char* ecl_kw_get_header ( ecl_kw )")
cfunc.fprintf_grdecl             = cwrapper.prototype("void ecl_kw_fprintf_grdecl( ecl_kw , FILE )")

