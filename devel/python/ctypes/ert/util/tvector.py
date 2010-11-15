import sys
import ctypes
import libutil
from   ert.cwrap.cwrap       import *
from   ert.util.cfile        import CFILE



class TVector(object):
    
    @classmethod
    def strided_copy( cls , obj , slice ):
        """Used to support sliced lookup"""
        start  = 0
        stop   = obj.size - 1
        stride = 1
        if slice.start:
            start = slice.start
        if slice.stop:
            stop = slice.stop
        if slice.step:
            stride = slice.step

        new_obj = TVector.__new__( cls )
        new_obj.c_ptr = cls.cstrided_copy( obj  , start , stop , stride )
        new_obj.data_owner = True
        return new_obj
    

    @classmethod
    def __copy__( cls , obj ):
        new_obj = TVector.__new__( cls )
        new_obj.c_ptr = cls.alloc_copy( obj )
        new_obj.data_owner = True
        return new_obj
    

    def __new__( cls ):
        obj = object.__new__( cls )
        obj.c_ptr = None
        return obj


    @classmethod
    def ref( cls , c_ptr , parent ):
        obj = cls( )
        obj.c_ptr      = c_ptr
        obj.data_owner = False
        obj.parent     = parent
        return obj

    
    def copy( self ):
        new = self.__copy__( self )  # Invoking the class method
        return new


    def __deepcopy__(self , memo):
        new = self.copy(  )
        return new

    def __init__( self , default_value = 0 , c_ptr = None):
        # Default initializer allocates a new instance from the C layer.
        init_size  = 0
        self.c_ptr = self.alloc( init_size , default_value )
        self.data_owner = True

    def from_param( self ):
        return self.c_ptr


    def __del__(self):
        if self.data_owner:
            self.free( self )

    def __getitem__(self, index ):
        if isinstance( index , types.IntType):
            length = self.__len__()
            if index < 0 or index >= length:
                raise IndexError
            else:
                return self.iget( self , index )
        elif isinstance( index , types.SliceType ):
            return self.strided_copy( self , index )
        else:
            raise TypeError("Index should be integer type")
        
    def __setitem__( self , index , value ):
        if isinstance( index , types.IntType):
            self.iset( self, index , value )
        else:
            raise TypeError("Index should be integer type")

    def __len__(self):
        return self.get_size( self )

    def printf( self , fmt = None , name = None , stream = sys.stdout ):
        cfile = CFILE( stream )
        if not fmt:
            fmt = self.def_fmt
        self.fprintf(self , cfile , name , fmt)

    @property
    def size( self ):
        return self.__len__()

    def append( self , value ):
        self.cappend( self , value )

    def del_block( self , index , block_size ):
        self.idel_block( self , index , block_size )

    def sort( self ):
        self.csort( self )

    def rsort( self ):
        self.crsort( self )

    def clear(self):
        self.cclear( self )

    def safe_iget( self , index):
        return self.csafe_iget( self , index )

    def set_read_only( self , read_only ):
        self.set_read_only( self , read_only )

    def get_read_only( self ):
        return self.get_read_only( self )
        
    read_only = property( get_read_only , set_read_only )



#################################################################


class DoubleVector(TVector):
    initialized = False

    def __new__( cls , *arglist ):
        if not cls.initialized:
            cls.csort         = cfunc.double_vector_sort
            cls.crsort        = cfunc.double_vector_rsort
            cls.alloc         = cfunc.double_vector_alloc
            cls.alloc_copy    = cfunc.double_vector_alloc_copy
            cls.free          = cfunc.double_vector_free
            cls.get_size      = cfunc.double_vector_size
            cls.iget          = cfunc.double_vector_iget
            cls.iset          = cfunc.double_vector_iset
            cls.fprintf       = cfunc.double_vector_fprintf
            cls.cappend       = cfunc.double_vector_append
            cls.idel_block    = cfunc.double_vector_idel_block            
            cls.cclear        = cfunc.double_vector_reset
            cls.cstrided_copy = cfunc.double_vector_strided_copy
            cls.csafe_iget    = cfunc.double_vector_safe_iget
            cls.set_read_only = cfunc.double_vector_set_read_only
            cls.get_read_only = cfunc.double_vector_get_read_only
            cls.def_fmt       = "%g"
            cls.initialized = True

        obj = TVector.__new__( cls )
        return obj
    


class IntVector(TVector):
    initialized = False
    
    def __new__( cls , *arglist ):
        if not cls.initialized:
            cls.csort         = cfunc.int_vector_sort
            cls.crsort         = cfunc.int_vector_rsort
            cls.alloc         = cfunc.int_vector_alloc
            cls.alloc_copy    = cfunc.int_vector_alloc_copy
            cls.free          = cfunc.int_vector_free
            cls.get_size      = cfunc.int_vector_size
            cls.iget          = cfunc.int_vector_iget
            cls.iset          = cfunc.int_vector_iset
            cls.fprintf       = cfunc.int_vector_fprintf
            cls.cappend       = cfunc.int_vector_append
            cls.idel_block    = cfunc.int_vector_idel_block            
            cls.cclear        = cfunc.int_vector_reset
            cls.cstrided_copy = cfunc.int_vector_strided_copy
            cls.csafe_iget    = cfunc.int_vector_safe_iget
            cls.set_read_only = cfunc.int_vector_set_read_only
            cls.get_read_only = cfunc.int_vector_get_read_only
            cls.def_fmt       = "%d"
            cls.initialized = True

        obj = TVector.__new__( cls )
        return obj
    
#################################################################

CWrapper.registerType( "double_vector" , DoubleVector )
CWrapper.registerType( "int_vector"    , IntVector )

cwrapper = CWrapper( libutil.lib )
cfunc    = CWrapperNameSpace("tvector")

cfunc.double_vector_alloc            = cwrapper.prototype("long   double_vector_alloc( int , double )")
cfunc.double_vector_alloc_copy       = cwrapper.prototype("long   double_vector_alloc_copy( double_vector )")
cfunc.double_vector_strided_copy     = cwrapper.prototype("long   double_vector_alloc_strided_copy( double_vector , int , int , int)")
cfunc.double_vector_free             = cwrapper.prototype("void   double_vector_free( double_vector )")
cfunc.double_vector_iget             = cwrapper.prototype("double double_vector_iget( double_vector , int )")
cfunc.double_vector_safe_iget        = cwrapper.prototype("double double_vector_safe_iget( int_vector , int )")
cfunc.double_vector_iset             = cwrapper.prototype("double double_vector_iset( double_vector , int , double)")
cfunc.double_vector_size             = cwrapper.prototype("int    double_vector_size( double_vector )")
cfunc.double_vector_append           = cwrapper.prototype("void   double_vector_append( double_vector , double )") 
cfunc.double_vector_idel_block       = cwrapper.prototype("void   double_vector_idel_block( double_vector , int , int )") 
cfunc.double_vector_fprintf          = cwrapper.prototype("void   double_vector_fprintf( double_vector , FILE , char* , char*)")
cfunc.double_vector_sort             = cwrapper.prototype("void   double_vector_sort( double_vector )") 
cfunc.double_vector_rsort            = cwrapper.prototype("void   double_vector_rsort( double_vector )") 
cfunc.double_vector_reset            = cwrapper.prototype("void   double_vector_reset( double_vector )") 
cfunc.double_vector_get_read_only    = cwrapper.prototype("bool   double_vector_set_read_only( double_vector )") 
cfunc.double_vector_set_read_only    = cwrapper.prototype("void   double_vector_set_read_only( double_vector , bool )") 

cfunc.int_vector_alloc_copy          = cwrapper.prototype("long int_vector_alloc_copy( int_vector )")
cfunc.int_vector_alloc               = cwrapper.prototype("long   int_vector_alloc( int , int )")
cfunc.int_vector_strided_copy        = cwrapper.prototype("long   int_vector_alloc_strided_copy( int_vector , int , int , int)")
cfunc.int_vector_free                = cwrapper.prototype("void   int_vector_free( int_vector )")
cfunc.int_vector_iget                = cwrapper.prototype("int    int_vector_iget( int_vector , int )")
cfunc.int_vector_safe_iget           = cwrapper.prototype("int    int_vector_safe_iget( int_vector , int )")
cfunc.int_vector_iset                = cwrapper.prototype("int    int_vector_iset( int_vector , int , int)")
cfunc.int_vector_size                = cwrapper.prototype("int    int_vector_size( int_vector )")
cfunc.int_vector_append              = cwrapper.prototype("void   int_vector_append( int_vector , int )") 
cfunc.int_vector_idel_block          = cwrapper.prototype("void   int_vector_idel_block( int_vector , int , int )") 
cfunc.int_vector_fprintf             = cwrapper.prototype("void   int_vector_fprintf( int_vector , FILE , char* , char*)")
cfunc.int_vector_sort                = cwrapper.prototype("void   int_vector_sort( int_vector )") 
cfunc.int_vector_rsort               = cwrapper.prototype("void   int_vector_rsort( int_vector )") 
cfunc.int_vector_reset               = cwrapper.prototype("void   int_vector_reset( int_vector )") 
cfunc.int_vector_set_read_only       = cwrapper.prototype("void   int_vector_set_read_only( int_vector , bool )") 
cfunc.int_vector_get_read_only       = cwrapper.prototype("bool   int_vector_get_read_only( int_vector )") 
