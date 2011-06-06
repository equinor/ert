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
    """
    Create and insert enum values as integer constants.

    This function is based on iteratively querying the C library for
    symbol name and enum value for an enum. The @lib argument should
    be a ctypes libhandle, i.e. the return value from a CDLL()
    call. The @func_name argument should be the name (i.e. a string)
    of a C-function in the library which can provide symbol name and
    numerical value for the elements in an enum.

    For an enum defition like:

       enum my_enum {
          INVALID = 0,
          VALUE1  = 1,
          VALUE2  = 2
       }

    The @func_name function in @lib should return

         'INVALID' , 0
         'VALUE1'  , 1
         'VALUE2'  , 2

    for index arguments 0,1 and 2 respectively.  The symbols are
    installed in the dictionary @dict[1], so after the enum above has
    been internalized the dict scope will have symols like:

        INVALID = 0
        VALUE1  = 1
        VALUE2  = 2
        
    The @dict dictionary should typically be the globals() dictionary
    from the calling scope.
    """

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
    return enum
        

    
