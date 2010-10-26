import sys
import ctypes
import libutil
from   ert.cwrap.cwrap       import *


class Hash:

    def __init__( self ):
        self.c_ptr = cfunc.hash_alloc()
        
    def __del__( self ):
        cfunc.hash_del( self )
        
    def __getitem__(self , key):
        pass


CWrapper.registerType( "hash" , Hash )
cwrapper         = CWrapper( libutil.lib )
cfunc            = CWrapperNameSpace("hash")

cfunc.hash_alloc = cwrapper.prototype("long hash_alloc( )")
cfunc.hash_free  = cwrapper.prototype("void hash_free( hash )")
cfunc.hash_get   = cwrapper.prototype("long hash_get(hash , char*)")
