#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'cfile.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details. 


import time
import datetime
import ctypes
import ert.util.pycfile
from   ert.util.pycfile      import pycfile       
from   ert.cwrap.cwrap       import *


# This purpose of this class is to wrap the underlying FILE * pointer
# of Python filehandle to be able to call C functions which expect
# FILE * input with a Python filehandle.
#
# The class needs support from the small C based extension module
# pycfile which will get a Python filehandle and return the underlying
# FILE * pointer value - this is stored in the c_ptr attribute.
#
# Observe that this small class is slightly different from the other
# thin python classes in that the from_param() method does not contain
# a ctypes.c_void_p() cast - quite frankly I can not explain why.


class CFILE:
    def __init__( self , py_file ):
        self.c_ptr   =  ctypes.c_void_p( pycfile( py_file ) )
        self.py_file =  py_file

    def from_param( self ):
        return self.c_ptr     

    def __del__(self):
        pass



cwrapper = CWrapper( None ) 
cwrapper.registerType( "FILE" , CFILE )
