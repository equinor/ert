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
#include <ert/enkf/ert_test_context.h>




void test_blockdata( enkf_main_type * enkf_main , const char * obs_key) {
  enkf_obs_type * enkf_obs = enkf_main_get_obs( enkf_main );
  obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key);
  int report_step = obs_vector_get_next_active_step( obs_vector , -1 );
  {
    enkf_plot_blockdata_type * block_data = enkf_plot_blockdata_alloc( obs_vector );
    enkf_fs_type * fs = enkf_main_get_fs( enkf_main );
    block_obs_type * block_obs = obs_vector_iget_node( obs_vector , report_step);

    test_assert_true( enkf_config_node_is_instance( obs_vector_get_config_node( obs_vector )));
    enkf_plot_blockdata_load( block_data , fs , report_step , FORECAST , NULL );
    test_assert_int_equal( enkf_main_get_ensemble_size( enkf_main ) , enkf_plot_blockdata_get_size( block_data ));
    {
      enkf_plot_blockvector_type * vector = enkf_plot_blockdata_iget( block_data , 0 );
      test_assert_true( enkf_plot_blockvector_is_instance( vector ));
      test_assert_int_equal( block_obs_get_size( block_obs ) , enkf_plot_blockvector_get_size( vector ));
      
    }

    enkf_plot_blockdata_free( block_data );
  }
}



int main( int argc , char ** argv) {
  const char * config_file = argv[1];
  util_install_signals();
  ert_test_context_type * test_context = ert_test_context_alloc("BLOCKDATA" , config_file , NULL );
  enkf_main_type * enkf_main = ert_test_context_get_main( test_context );
  
  test_blockdata( enkf_main , "RFT2");
  test_blockdata( enkf_main , "RFT5");
  
  ert_test_context_free( test_context );
  exit(0);
}
