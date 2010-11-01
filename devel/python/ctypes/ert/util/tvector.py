import sys
import ctypes
import libutil
from   ert.cwrap.cwrap       import *
from   ert.util.cfile        import CFILE

INT_TYPE    = 1
DOUBLE_TYPE = 2


# Should have class based constructor

class TVector:
    ##
    ##@classmethod
    ##def __new__( cls , vtype ):
    ##    obj = object.__new__( cls )
    ##    obj.__init( vtype )
    ##    obj.c_ptr = None
    ##    return obj
    ##
    ##@classmethod
    ##def new( cls , vtype , init_size = 0, default_value = 0):
    ##    obj = TVector.__new__( vtype )
    ##    obj.c_ptr = obj.alloc( init_size , default_value )
    ##    return obj
    ##
    ##@classmethod
    ##def copy( cls , src):
    ##    pass
    ##
    ##
    
    @classmethod
    def strided_copy( cls , src , slice ):
        obj = cls()
        obj.__init( type )
    
        start  = 0
        stop   = self.size - 1
        stride = 1
        if slice.start:
            start = slice.start
        if slice.stop:
            stop = slice.stop
        if slice.step:
            stride = slice.step
            
        obj.c_ptr = obj.cstrided_copy( src , start , stop , stride )
        return obj
    

    def __init__( self , type , init_size , default_value):
        self.type = type
        if type == INT_TYPE:
            self.csort         = cfunc.int_vector_sort
            self.alloc         = cfunc.int_vector_alloc
            self.free          = cfunc.int_vector_free
            self.get_size      = cfunc.int_vector_size
            self.iget          = cfunc.int_vector_iget
            self.iset          = cfunc.int_vector_iset
            self.fprintf       = cfunc.int_vector_fprintf
            self.cappend       = cfunc.int_vector_append
            self.cstrided_copy = cfunc.int_vector_strided_copy
            self.idel_block    = cfunc.int_vector_idel_block
            self.cclear        = cfunc.int_vector_reset
            self.def_fmt       = "%d"
        elif type == DOUBLE_TYPE:
            self.csort         = cfunc.double_vector_sort
            self.alloc         = cfunc.double_vector_alloc
            self.free          = cfunc.double_vector_free
            self.get_size      = cfunc.double_vector_size
            self.iget          = cfunc.double_vector_iget
            self.iset          = cfunc.double_vector_iset
            self.fprintf       = cfunc.double_vector_fprintf
            self.cappend       = cfunc.double_vector_append
            self.idel_block    = cfunc.double_vector_idel_block            
            self.cclear        = cfunc.double_vector_reset
            self.cstrided_copy = cfunc.double_vector_strided_copy
            self.def_fmt       = "%g"
        else:
            sys.exit("Error")
        
        self.c_ptr = self.alloc( init_size , default_value )


        
    def from_param( self ):
        return self.c_ptr

    def __del__(self):
        self.free( self )

    def __getitem__(self, index ):
        if isinstance( index , types.IntType):
            length = self.__len__()
            if index < 0 or index >= length:
                raise IndexError
            else:
                return self.iget( self , index )
        elif isinstance( index , types.SliceType ):
            return self.strided_copy( index )
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

    def clear(self):
        self.cclear( self )

#################################################################


class DoubleVector( TVector ):
    def __init__( self , type , init_size = 0, default_value = 0):
        TVector.__init__( self , DOUBLE_TYPE , init_size , default_value )


class IntVector(TVector):
    def __init__( self , init_size = 0 , default_value = 0):
        TVector.__init__( self , INT_TYPE , init_size , default_value )




CWrapper.registerType( "double_vector" , DoubleVector )
CWrapper.registerType( "int_vector"    , IntVector )

cwrapper = CWrapper( libutil.lib )
cfunc    = CWrapperNameSpace("tvector")

cfunc.double_vector_alloc            = cwrapper.prototype("long double_vector_alloc( int , double )")
cfunc.double_vector_strided_copy     = cwrapper.prototype("long   double_vector_alloc_strided_copy( double_vector , int , int , int)")
cfunc.double_vector_free             = cwrapper.prototype("void double_vector_free( double_vector )")
cfunc.double_vector_iget             = cwrapper.prototype("double double_vector_iget( double_vector , int )")
cfunc.double_vector_iset             = cwrapper.prototype("double double_vector_iset( double_vector , int , double)")
cfunc.double_vector_size             = cwrapper.prototype("int    double_vector_size( double_vector )")
cfunc.double_vector_append           = cwrapper.prototype("void   double_vector_append( double_vector , double )") 
cfunc.double_vector_idel_block       = cwrapper.prototype("void   double_vector_idel_block( double_vector , int , int )") 
cfunc.double_vector_fprintf          = cwrapper.prototype("void   double_vector_fprintf( double_vector , FILE , char* , char*)")
cfunc.double_vector_sort             = cwrapper.prototype("void   double_vector_sort( double_vector )") 
cfunc.double_vector_reset            = cwrapper.prototype("void   double_vector_reset( double_vector )") 


cfunc.int_vector_alloc               = cwrapper.prototype("long   int_vector_alloc( int , int )")
cfunc.int_vector_strided_copy        = cwrapper.prototype("long   int_vector_alloc_strided_copy( int_vector , int , int , int)")
cfunc.int_vector_free                = cwrapper.prototype("void   int_vector_free( int_vector )")
cfunc.int_vector_iget                = cwrapper.prototype("int    int_vector_iget( int_vector , int )")
cfunc.int_vector_iset                = cwrapper.prototype("int    int_vector_iset( int_vector , int , int)")
cfunc.int_vector_size                = cwrapper.prototype("int    int_vector_size( int_vector )")
cfunc.int_vector_append              = cwrapper.prototype("void   int_vector_append( int_vector , int )") 
cfunc.int_vector_idel_block          = cwrapper.prototype("void   int_vector_idel_block( int_vector , int , int )") 
cfunc.int_vector_fprintf             = cwrapper.prototype("void   int_vector_fprintf( int_vector , FILE , char* , char*)")
cfunc.int_vector_sort                = cwrapper.prototype("void   int_vector_sort( int_vector )") 
cfunc.int_vector_reset               = cwrapper.prototype("void   int_vector_reset( int_vector )") 
