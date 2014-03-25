/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
   The file 'enkf_obs_vector_fs.c' is part of ERT - Ensemble based
   Reservoir Tool.
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/


#include <stdlib.h>
#include <stdio.h>

#include <ert/enkf/enkf_obs.h>
#include <ert/enkf/ert_test_context.h>
#include <ert/enkf/meas_data.h>
#include <ert/enkf/obs_data.h>
#include <ert/enkf/local_obsset.h>
#include <ert/enkf/obs_vector.h>

#include <ert/util/test_util.h>


void test_valid_obs_vector( enkf_main_type * enkf_main , const char * obs_key) {
  enkf_fs_type * fs = enkf_main_get_fs(enkf_main);
  enkf_obs_type * enkf_obs = enkf_main_get_obs( enkf_main );
  obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );
  bool_vector_type * active_mask = bool_vector_alloc( enkf_main_get_ensemble_size( enkf_main ) , true);
  
  test_assert_true( obs_vector_has_data( obs_vector , active_mask , fs ));
  bool_vector_free( active_mask );
}


void test_invalid_obs_vector( enkf_main_type * enkf_main , const char * obs_key) {
  enkf_fs_type * fs = enkf_main_get_fs(enkf_main);
  enkf_obs_type * enkf_obs = enkf_main_get_obs( enkf_main );
  obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );
  bool_vector_type * active_mask = bool_vector_alloc( enkf_main_get_ensemble_size( enkf_main ) , true);
  
  test_assert_false( obs_vector_has_data( obs_vector , active_mask , fs ));
  bool_vector_free( active_mask );
}



int main(int argc , char ** argv) {
  const char * config_file = argv[1];
  ert_test_context_type * context = ert_test_context_alloc( "OBS_VECTOR_FS" , config_file , NULL);
  enkf_main_type * enkf_main = ert_test_context_get_main( context );
  
  {
    test_valid_obs_vector( enkf_main , "WWCT:OP_3");
    test_invalid_obs_vector( enkf_main , "GOPT:OP");
  }
  ert_test_context_free( context );
}

