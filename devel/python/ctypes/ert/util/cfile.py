import time
import datetime
import ctypes
from   ert.util.pycfile      import pycfile
from   ert.cwrap.cwrap       import *


# This purpose of this class is to wrap the underlying FILE * pointer
# of Python filehandle to be able to call C functions which expect
# FILE * input with a Python filehandle.
#
# The class needs support from the small C based extension module
# pycfile which will get a Python filehandle and return the underlying
# FILE * pointer value - this is stored in the c_ptr attribute.

class CFILE:
    def __init__( self , py_file ):
        self.c_ptr   =  ctypes.c_long( pycfile( py_file ) )
        self.py_file =  py_file

    def from_param( self ):
        return self.c_ptr

    def __del__(self):
        pass




cwrapper = CWrapper( None ) 
cwrapper.registerType( "FILE" , CFILE )
