import ctypes
from   ert.cwrap.cwrap       import *


class DoubleVector:
    def __init__( self , init_size = 0 , default_value = 0):
        self.c_ptr = cfunc.double_vector_alloc( init_size , default_value )
        
    def from_param( self ):
        return self.c_ptr

    def __del__(self):
        cfunc.double_vector_free( self )

    def __getitem__(self, index ):
        if isinstance( index , types.IntType):
            length = self.__len__()
            if index < 0 or index >= length:
                raise IndexError
            else:
                return cfunc.double_vector_iget( self , index )
        else:
            raise TypeError("Index should be integer type")

    def __setitem__( self , index , value ):
        cfunc.double_vector_iset( self, index , value )

    def __len__(self):
        return cfunc.double_vector_size( self )

    @property
    def size( self ):
        return cfunc.double_vector_size( self )
    
    def append( self , value ):
        cfunc.double_vector_append( self , value )
    

ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
libutil = ctypes.CDLL("libutil.so" , ctypes.RTLD_GLOBAL)

CWrapper.registerType( "double_vector" , DoubleVector )

cwrapper = CWrapper( libutil )
cfunc    = CWrapperNameSpace("tvector")
cfunc.double_vector_alloc   = cwrapper.prototype("long double_vector_alloc( int , double )")
cfunc.double_vector_free    = cwrapper.prototype("void double_vector_free( double_vector )")
cfunc.double_vector_iget    = cwrapper.prototype("double double_vector_iget( double_vector , int )")
cfunc.double_vector_iset    = cwrapper.prototype("double double_vector_iset( double_vector , int , double)")
cfunc.double_vector_size    = cwrapper.prototype("int    double_vector_size( double_vector )")
cfunc.double_vector_append  = cwrapper.prototype("void   double_vector_append( double_vector , double )") 
