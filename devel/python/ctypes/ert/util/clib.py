#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'clib.py' is part of ERT - Ensemble based Reservoir Tool. 
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
import os

# The shared libraries typically exist under several different names,
# with different level of version detail. Unfortunately the same
# library can exist under different names on the different Statoil
# computers, to support this the load function can get several
# arguments like:
#
#    load("libz.so" , "libz.so.1" , "libz.so.1.2.1.2" , "libZ-fucker.so")
#
# Will return a handle to the first successfull load, and raise
# ImportError if none of the loads succeed.

def load( *lib_list ):
    dll = None
    
    for lib in lib_list:
        try:
            dll = ctypes.CDLL( lib , ctypes.RTLD_GLOBAL )
            return dll
        except:
            pass
    
    error_msg = "Sorry - failed to load shared library:%s\n\nTried in: " % lib_list[0]
    path_list = os.getenv("LD_LIBRARY_PATH").split(":")
    for path in path_list:
        error_msg += path + "\n          "

    raise ImportError( error_msg )
