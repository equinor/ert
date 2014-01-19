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


void test_create() {
  ecl_grid_type * grid = ecl_grid_alloc_rectangular(10 , 10 , 10 , 1 , 1, 1, NULL);
  field_config_type * field_config = field_config_alloc_empty("FIELD" , grid , NULL );
  block_obs_type * block_obs = block_obs_alloc("Obs" , field_config , grid );

  {
    enkf_plot_blockdata_type * block_data = enkf_plot_blockdata_alloc( block_obs );
    test_assert_true( enkf_plot_blockdata_is_instance( block_data ));
    test_assert_int_equal( 0 , enkf_plot_blockdata_get_size( block_data ));
    enkf_plot_blockdata_free( block_data );
  }
  
  block_obs_free( block_obs );
  field_config_free( field_config );
  ecl_grid_free( grid );
}




int main( int argc , char ** argv) {
  test_create();
  exit(0);
}
