#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'cenum.py' is part of ERT - Ensemble based Reservoir Tool. 
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


import ctypes
import sys

def create_enum( lib, func_name , enum_name , dict):
    try:
        func = getattr( lib , func_name )
    except AttributeError:
        sys.exit("Could not find enum description function:%s - can not load enum:%s." % (func_name , enum_name))

    func.restype = ctypes.c_char_p
    func.argtypes = [ ctypes.c_int , ctypes.POINTER(ctypes.c_int) ]
    enum = {}
    index = 0
    while True:
        value = ctypes.c_int()
        name = func( index , ctypes.byref( value ))
        if name:
            dict[ name ] = value.value
            enum[ name ] = value.value
            index += 1
        else:
            break
    dict[ enum_name ] = enum
        

    
