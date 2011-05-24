#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl_util.py' is part of ERT - Ensemble based Reservoir Tool. 
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
"""
Constants from the header ecl_util.h - some stateless functions.

This module does not contain any class definitions; it mostly consists
of enum definitions/values which are copied verbatim from the
ecl_util.h header. Observe that the the numerical values given here
must be KEPT IN SYNC WITH ecl_util.h MANUALLY. 

In addition to the enum definitions there are a few stateless
functions from ecl_util.c which are not bound to any class type.
"""

from    ert.cwrap.cwrap       import *
import  libecl


## Enum ecl_file_enum from ecl_util.h
ECL_OTHER_FILE           = 0   
ECL_RESTART_FILE         = 1   
ECL_UNIFIED_RESTART_FILE = 2   
ECL_SUMMARY_FILE         = 4   
ECL_UNIFIED_SUMMARY_FILE = 8   
ECL_SUMMARY_HEADER_FILE  = 16  
ECL_GRID_FILE            = 32  
ECL_EGRID_FILE           = 64  
ECL_INIT_FILE            = 128 
ECL_RFT_FILE             = 256 
ECL_DATA_FILE            = 512 


# ecl_phase_enum from ecl_util.h
ECL_OIL_PHASE   = 1
ECL_GAS_PHASE   = 2
ECL_WATER_PHASE = 4


# Enum defintion from ecl_util.h
ECL_CHAR_TYPE   = 0
ECL_FLOAT_TYPE  = 1
ECL_DOUBLE_TYPE = 2
ECL_INT_TYPE    = 3
ECL_BOOL_TYPE   = 4
ECL_MESS_TYPE   = 5

def get_num_cpu( datafile ):
    """
    Parse ECLIPSE datafile and determine how many CPUs are needed.

    Will look for the "PARALLELL" keyword, and then read off the
    number of CPUs required. Will return one if no PARALLELL keyword
    is found.
    """
    return cfunc.get_num_cpu( datafile )


def get_file_type( filename ):
    """
    Will inspect an ECLIPSE filename and return an integer type flag.
    """
    return cfunc.get_file_type( filename , None , None )


cwrapper             = CWrapper( libecl.lib )
cfunc                = CWrapperNameSpace("ecl_util")

cfunc.get_num_cpu    = cwrapper.prototype("int ecl_util_get_num_cpu( char* )")
cfunc.get_file_type  = cwrapper.prototype("int ecl_util_get_file_type( char* , bool* , int*)")
