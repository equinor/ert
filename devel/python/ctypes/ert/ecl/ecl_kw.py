from    ert.cwrap.cwrap       import *
import  types
import  numpy
import  fortio
import  libecl
from    ert.util.cfile   import CFILE



# Enum defintion from ecl_util.h1
ECL_CHAR_TYPE   = 0
ECL_FLOAT_TYPE  = 1
ECL_DOUBLE_TYPE = 2
ECL_INT_TYPE    = 3
ECL_BOOL_TYPE   = 4
ECL_MESS_TYPE   = 5


class EclKW(object):
    
    @classmethod
    def new( cls , name, size , type):
        obj = cls()
        obj.c_ptr      = cfunc.alloc_new( name , size , type )
        obj.data_owner = True
        obj.parent     = None
        obj.__init( )
        return obj


    @classmethod
    def ref( cls , c_ptr , parent ):
        obj = cls( )
        obj.c_ptr      = c_ptr
        obj.data_owner = False
        obj.parent     = parent
        obj.__init( )
        return obj
        

    @classmethod
    def copy( cls , src ):
        obj = cls( )
        obj.c_ptr = cfunc.copyc( src )
        obj.parent = None
        obj.data_owner = True
        obj.__init( )
        return obj
    
    
    @classmethod
    def grdecl_load( cls , file , kw , ecl_type = ECL_FLOAT_TYPE):
        cfile  = CFILE( file )
        c_ptr  = cfunc.load_grdecl( cfile , kw , ecl_type )
        if c_ptr:
            obj = cls( )
            obj.c_ptr = c_ptr
            obj.data_owner = True
            obj.parent = None
            obj.__init()
            return obj
        else:
            return None

    def ecl_kw_instance( self ):
        return True


    def __init(self):
        self.data_ptr   = None
        self.__type = cfunc.get_type( self )
        if self.__type == ECL_INT_TYPE:
            self.data_ptr = cfunc.int_ptr( self )
            self.dtype    = numpy.int32        
        elif self.__type == ECL_FLOAT_TYPE:
            self.data_ptr = cfunc.float_ptr( self )
            self.dtype    = numpy.float32
        elif self.__type == ECL_DOUBLE_TYPE:
            self.data_ptr = cfunc.double_ptr( self )
            self.dtype    = numpy.float64        
        else:
            # Iteration not supported ...
            self.data_ptr = None
            self.dtype    = None
            

    def from_param(self):
        return self.c_ptr

    def __len__( self ):
        return cfunc.get_size( self )

    
    def __del__(self):
        if self.data_owner:
            cfunc.free( self )


    def __deep_copy__(self , memo):
        ecl_kw = EclKW.copy( self )
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


    #################################################################
    

    def __IMUL__(self , factor , mul = True):
        if cfunc.assert_numeric( self ):
            if hasattr( factor , "ecl_kw_instance"):
                if cfunc.assert_binary( self, factor ):
                    if mul:
                        cfunc.imul( self , factor )
                    else:
                        cfunc.idiv( self , factor )
                else:
                    raise TypeError("Type mismatch")
            else:
                if not mul:
                    factor = 1.0 / factor
                    
                if self.__type == ECL_INT_TYPE:
                    if isinstance( factor , int ):
                        cfunc.scale_int( self , factor )
                    else:
                        raise TypeError("Type mismatch")
                else:
                    if isinstance( factor , int ) or isinstance( factor , float):
                        cfunc.scale_float( self , factor )
                    else:
                        raise TypeError("Only muliplication with scalar supported")
        else:
            raise TypeError("Not numeric type")
        
        return self
                

    def __IADD__(self , delta , add = True):
        if cfunc.assert_numeric( self ):
            if hasattr( delta , "ecl_kw_instance"):
                if cfunc.assert_binary( self, delta):
                    if add:
                        cfunc.iadd(self , delta )
                    else:
                        cfunc.isub( self , delta )
                else:
                    raise TypeError("Type / size mismatch")
            else:
                if add:
                    sign = 1
                else:
                    sign = -1

                if self.__type == ECL_INT_TYPE:
                    if isinstance( delta , int ):
                        cfunc.shift_int( self , delta * sign)
                    else:
                        raise TypeError("Type mismatch")
                else:
                    if isinstance( delta , int ) or isinstance( delta , float):
                        cfunc.shift_float( self , delta * sign ) # Will call the _float() or _double() function in the C layer.
                    else:
                        raise TypeError("Type mismatch")
        else:
            raise TypeError("Type / size mismatch")
        
        return self

    def __iadd__(self , delta):
        return self.__IADD__(delta , True )

    def __isub__(self , delta):
        return self.__IADD__(delta , False )

    def __imul__(self , delta):
        return self.__IMUL__(delta , True )

    def __idiv__(self , delta):
        return self.__IMUL__(delta , False )


    #################################################################
    
    
    def __add__(self , delta):
        copy = self.deep_copy()
        copy += delta
        return copy

    def __radd__(self, delta):
        return self.__add__( delta )

    def __sub__(self , delta):
        copy  = self.deep_copy()
        copy -= delta
        return copy

    def __rsub__( self , delta):
        return self.__sub__( delta ) * -1 
    
    def __mul__(self , factor):
        copy = self.deep_copy()
        copy *= factor
        return copy

    def __rmul__(self , factor):
        return self.__mul__( factor )
    
    def __div__(self , factor):
        copy = self.deep_copy()
        copy /= factor
        return copy
    
    # No __rdiv__()

    def assert_binary( self , other ):
        return cfunc.assert_binary( self , other )

    #################################################################
        
    def assign(self , value , mask = None , force_active = False):
        if cfunc.assert_numeric( self ):
            if hasattr( value , "ecl_kw_instance"):
                if mask:
                    mask.copy_kw( self , value , force_active)
                else:
                    if self.assert_binary( value ):
                        cfunc.copy_data( self , value)
                    else:
                        raise TypeError("Type / size mismatch")
            else:
                if mask:
                    mask.set_kw( self , value , force_active )
                else:
                    if self.__type == ECL_INT_TYPE:
                        if isinstance( factor , int ):
                            cfunc.set_int( self , value )
                        else:
                            raise TypeError("Type mismatch")
                    else:
                        if isinstance( value , int ) or isinstance( value, float):
                            cfunc.set_float( self , value )
                        else:
                            raise TypeError("Only muliplication with scalar supported")

                        
    def add( self , other , mask = None , force_active = False):
        if mask:
            mask.iadd_kw( self , other , force_active )
        else:
            return self.__iadd__( other )
        
    def sub(self , other , mask = None , force_active = False):
        if mask:
            mask.isub_kw(  self , other , force_active )
        else:
            return self.__isub__( other )

    def mul(self , other , mask = None , force_active = False):
        if mask:
            mask.imul_kw(  self , other , force_active )
        else:
            return self.__imul__( other )

    def div(self , other , mask = None , force_active = False):
        if mask:
            mask.idiv_kw(  self , other , force_active )
        else:
            return self.__idiv__( other )

    def apply( self , func , arg = None , mask = None , force_active = False):
        """Will apply the function @func on the keyword - inplace.

        The function @func should take a scalar value from the ecl_kw
        vector as input, and return a scalar value of the same type;
        optionally you can supply a second argument with the @arg
        attribute:

          def cutoff( x , limit):
              if x > limit:
                 return x
              else:
                 return 0


          kw.apply( math.sin )
          kw.apply( cutoff , arg = 0.10 )

        
        It is possible to supply a EclRegion instance via the @mask
        attribute.
        """
        if mask:
            active_list = mask.kw_list( self , force_active )
            if arg:
                for i in range(active_list.size):
                    self.data_ptr[i] = func( self.data_ptr[i] , arg)
            else:
                for i in range(active_list.size):
                    self.data_ptr[i] = func( self.data_ptr[i] )
        else:
            if arg:
                for i in range(self.size):
                    self.data_ptr[i] = func( self.data_ptr[i] , arg)
            else:
                for i in range(self.size):
                    self.data_ptr[i] = func( self.data_ptr[i] )
                    




    #################################################################

    def deep_copy( self ):
        ecl_kw = self.__deep_copy__( {} )
        return ecl_kw

    @property
    def size(self):
        return cfunc.get_size( self )
    
    def set_name( self , name ):
        cfunc.set_header( self , name )

    def get_name( self ):
        return cfunc.get_header( self )
        
    name = property( get_name , set_name )

    @property
    def type( self ):
        # enum ecl_type_enum from ecl_util.h
        if self.__type == ECL_CHAR_TYPE:
            return "CHAR"
        if self.__type == ECL_FLOAT_TYPE:
            return "REAL"
        if self.__type == ECL_DOUBLE_TYPE:
            return "DOUB"
        if self.__type == ECL_INT_TYPE:
            return "INTE"
        if self.__type == ECL_BOOL_TYPE:
            return "BOOL"
        if self.__type == ECL_MESS_TYPE:
            return "MESS"


    @property    
    def min_max( self ):
        if self.__type == ECL_FLOAT_TYPE:
            min = ctypes.c_float()
            max = ctypes.c_float()
            cfunc.max_min_float( self , ctypes.byref( max ) , ctypes.byref( min ))
        elif self.__type == ECL_DOUBLE_TYPE:
            min = ctypes.c_double()
            max = ctypes.c_double()
            cfunc.max_min_double( self , ctypes.byref( max ) , ctypes.byref( min ))
        elif self.__type == ECL_INT_TYPE:
            min = ctypes.c_int()
            max = ctypes.c_int()
            cfunc.max_min_int( self , ctypes.byref( max ) , ctypes.byref( min ))
        else:
            print self.type
            return None
        return (min.value , max.value)


    @property
    def max( self ):
        min_max = self.min_max
        if min_max:
            return min_max[1]
        else:
            return None

    
    @property
    def min( self ):
        min_max = self.min_max
        if min_max:
            return min_max[0]
        else:
            return None

    
    @property
    def numeric(self):
        if self.__type == ECL_FLOAT_TYPE:
            return True
        if self.__type == ECL_DOUBLE_TYPE:
            return True
        if self.__type == ECL_INT_TYPE:
            return True
        return False

    
    @property
    def __type__( self ):
        return self.__type


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
cfunc.set_header                 = cwrapper.prototype("void  ecl_kw_set_header_name ( ecl_kw , char*)")
cfunc.fprintf_grdecl             = cwrapper.prototype("void ecl_kw_fprintf_grdecl( ecl_kw , FILE )")
cfunc.alloc_new                  = cwrapper.prototype("long ecl_kw_alloc( char* , int , int )")
cfunc.load_grdecl                = cwrapper.prototype("long ecl_kw_fscanf_alloc_grdecl_dynamic( FILE , char* , int )")
cfunc.fseek_grdecl               = cwrapper.prototype("bool ecl_kw_grdecl_fseek_kw(char* , bool, bool , FILE )")

cfunc.iadd                       = cwrapper.prototype("void ecl_kw_inplace_add( ecl_kw , ecl_kw )")
cfunc.imul                       = cwrapper.prototype("void ecl_kw_inplace_mul( ecl_kw , ecl_kw )")
cfunc.idiv                       = cwrapper.prototype("void ecl_kw_inplace_div( ecl_kw , ecl_kw )")
cfunc.isub                       = cwrapper.prototype("void ecl_kw_inplace_sub( ecl_kw , ecl_kw )")

cfunc.assert_binary              = cwrapper.prototype("bool ecl_kw_assert_binary_numeric( ecl_kw , ecl_kw )")
cfunc.scale_int                  = cwrapper.prototype("void ecl_kw_scale_int( ecl_kw , int )")
cfunc.scale_float                = cwrapper.prototype("void ecl_kw_scale_float_or_double( ecl_kw , double )")
cfunc.shift_int                  = cwrapper.prototype("void ecl_kw_shift_int( ecl_kw , int )")
cfunc.shift_float                = cwrapper.prototype("void ecl_kw_shift_float_or_double( ecl_kw , double )")
cfunc.assert_numeric             = cwrapper.prototype("bool ecl_kw_assert_numeric( ecl_kw )")
cfunc.copy_data                  = cwrapper.prototype("void ecl_kw_memcpy_data( ecl_kw , ecl_kw )")
cfunc.set_int                    = cwrapper.prototype("void ecl_kw_scalar_set_int( ecl_kw , int )")
cfunc.set_float                  = cwrapper.prototype("void ecl_kw_scalar_set_float_or_double( ecl_kw , double )")

cfunc.max_min_int                = cwrapper.prototype("void ecl_kw_max_min_int( ecl_kw , int* , int*)")
cfunc.max_min_float              = cwrapper.prototype("void ecl_kw_max_min_float( ecl_kw , float* , float*)")
cfunc.max_min_double             = cwrapper.prototype("void ecl_kw_max_min_double( ecl_kw , double* , double*)")

