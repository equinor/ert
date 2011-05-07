#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl_grav.py' is part of ERT - Ensemble based Reservoir Tool. 
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



import libecl
from   ert.cwrap.cwrap       import *


class EclGrav:
    def __init__( self , grid_file , init_file ):
        self.c_ptr = cfunc.grav_alloc( grid_file , init_file )
        
    def __del__( self ):
        cfunc.free( self )

    def from_param( self ):
        return self.c_ptr

    def add_survey( self , survey_name , restart_file ):
        cfunc.add_survey( self , survey_name , restart_file )

    def add_survey_RPORV( self , survey_name , restart_file ):
        cfunc.add_survey_RPORV( self , survey_name , restart_file )

    def add_survey_PORMOD( self , survey_name , restart_file ):
        cfunc.add_survey_PORMOD( self , survey_name , restart_file )

    def add_survey_FIP( self , survey_name , restart_file ):
        cfunc.add_survey_FIP( self , survey_name , restart_file )
                
    def eval(self , base_survey , monitor_survey , pos):
        return cfunc.eval( self , base_survey , monitor_survey , pos[0] , pos[1] , pos[2])

    def new_std_density( self , phase_enum , default_density):
        cfunc.new_std_density( self , phase_enum , default_density )
        
    def add_std_density( self , phase_enum , pvtnum , density):
        cfunc.add_std_density( self , phase_enum , pvtnum , density )


# 2. Creating a wrapper object around the libecl library, 
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType( "ecl_grav" , EclGrav )


# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_grav")

cfunc.grav_alloc  = cwrapper.prototype("c_void_p   ecl_grav_alloc( char* , char* )")
cfunc.free        = cwrapper.prototype("void       ecl_grav_free( ecl_grav )")

# Return value ignored in the add_survey_xxx() functions:
cfunc.add_survey         = cwrapper.prototype("c_void_p  ecl_grav_add_survey( ecl_grav , char* , ecl_file )")
cfunc.add_survey_RPORV   = cwrapper.prototype("c_void_p  ecl_grav_add_survey_RPORV( ecl_grav , char* , ecl_file )")
cfunc.add_survey_PORMOD  = cwrapper.prototype("c_void_p  ecl_grav_add_survey_PORMOD( ecl_grav , char* , ecl_file )")
cfunc.add_survey_FIP  = cwrapper.prototype("c_void_p  ecl_grav_add_survey_FIP( ecl_grav , char* , ecl_file )")
cfunc.new_std_density    = cwrapper.prototype("void      ecl_grav_new_std_density( ecl_grav , int , double)")
cfunc.add_std_density    = cwrapper.prototype("void      ecl_grav_add_std_density( ecl_grav , int , int , double)")
cfunc.eval               = cwrapper.prototype("double    ecl_grav_eval( ecl_grav , char* , char* , double , double , double)")
