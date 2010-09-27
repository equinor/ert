import time
import datetime
import ctypes
from   ert.util.pycfile      import pycfile
from   ert.cwrap.cwrap       import *


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
