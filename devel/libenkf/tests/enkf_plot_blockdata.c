/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
   The file 'enkf_plot_blockdata.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ert/util/test_util.h>

#include <ert/ecl/ecl_grid.h>

#include <ert/enkf/block_obs.h>
#include <ert/enkf/enkf_plot_blockdata.h>
#include <ert/enkf/obs_vector.h>

void test_create() {
  ecl_grid_type * grid = ecl_grid_alloc_rectangular(10 , 10 , 10 , 1 , 1, 1, NULL);
  enkf_config_node_type * config_node = enkf_config_node_alloc_field( "FIELD" , grid , NULL , false);
  
  enkf_config_node_update_state_field( config_node , TRUNCATE_NONE , 0 , 0 );
  {
    obs_vector_type * obs_vector = obs_vector_alloc(BLOCK_OBS , "OBS" , config_node , 100);
    {
      enkf_plot_blockdata_type * block_data = enkf_plot_blockdata_alloc( obs_vector );
      test_assert_true( enkf_plot_blockdata_is_instance( block_data ));
      test_assert_int_equal( 0 , enkf_plot_blockdata_get_size( block_data ));
      enkf_plot_blockdata_free( block_data );
    }
    obs_vector_free( obs_vector );
  }
  enkf_config_node_free( config_node );
  ecl_grid_free( grid );
}




int main( int argc , char ** argv) {
  test_create();
  exit(0);
}
