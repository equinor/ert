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


def get_num_cpu( datafile ):
    return cfunc.get_num_cpu( datafile )


def get_file_type( filename ):
    return cfunc.get_file_type( filename , None , None )



cwrapper             = CWrapper( libecl.lib )
cfunc                = CWrapperNameSpace("ecl_util")
cfunc.get_num_cpu    = cwrapper.prototype("int ecl_util_get_num_cpu( char* )")
cfunc.get_file_type  = cwrapper.prototype("int ecl_util_get_file_type( char* , bool* , int*)")
