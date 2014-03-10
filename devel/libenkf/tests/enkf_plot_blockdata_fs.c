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




void test_blockdata( enkf_main_type * enkf_main , const char * obs_key , int report_step ) {
  enkf_obs_type * enkf_obs = enkf_main_get_obs( enkf_main );
  obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key);
  {
    enkf_plot_blockdata_type * block_data = enkf_plot_blockdata_alloc( obs_vector );
    enkf_fs_type * fs = enkf_main_get_fs( enkf_main );
    block_obs_type * block_obs = obs_vector_iget_node( obs_vector , report_step);

    test_assert_true( enkf_config_node_is_instance( obs_vector_get_config_node( obs_vector )));
    enkf_plot_blockdata_load( block_data , fs , report_step , FORECAST , NULL );
    test_assert_int_equal( enkf_main_get_ensemble_size( enkf_main ) , enkf_plot_blockdata_get_size( block_data ));

    {
      const double_vector_type * depth = enkf_plot_blockdata_get_depth( block_data );
      test_assert_double_equal( 1752.24998474 , double_vector_iget( depth , 0 ));
      test_assert_double_equal( 1757.88926697 , double_vector_iget( depth , 1 ));
      test_assert_double_equal( 1760.70924377 , double_vector_iget( depth , 2 ));
      if (report_step == 56)
        test_assert_double_equal( 1763.52885437 , double_vector_iget( depth , 3));
    }
    
    {
      enkf_plot_blockvector_type * vector = enkf_plot_blockdata_iget( block_data , 0 );
      test_assert_true( enkf_plot_blockvector_is_instance( vector ));
      test_assert_int_equal( block_obs_get_size( block_obs ) , enkf_plot_blockvector_get_size( vector ));
      
      
      if (report_step == 50) {
        test_assert_double_equal( 244.681655884 , enkf_plot_blockvector_iget( vector , 0 ));
        test_assert_double_equal( 245.217041016 , enkf_plot_blockvector_iget( vector , 1 ));
        test_assert_double_equal( 245.48500061  , enkf_plot_blockvector_iget( vector , 2 ));
      } else {
        test_assert_double_equal( 239.7550354   , enkf_plot_blockvector_iget( vector , 0 ));
        test_assert_double_equal( 240.290313721 , enkf_plot_blockvector_iget( vector , 1 ));
        test_assert_double_equal( 240.558197021 , enkf_plot_blockvector_iget( vector , 2 ));
        test_assert_double_equal( 240.825881958 , enkf_plot_blockvector_iget( vector , 3 ));
      }
    }

    {
      enkf_plot_blockvector_type * vector = enkf_plot_blockdata_iget( block_data , 9 );
      test_assert_true( enkf_plot_blockvector_is_instance( vector ));
      test_assert_int_equal( block_obs_get_size( block_obs ) , enkf_plot_blockvector_get_size( vector ));
      
      
      if (report_step == 50) {
        test_assert_double_equal( 238.702560425 , enkf_plot_blockvector_iget( vector , 0 ));
        test_assert_double_equal( 239.237838745 , enkf_plot_blockvector_iget( vector , 1 ));
        test_assert_double_equal( 239.505737305 , enkf_plot_blockvector_iget( vector , 2 ));
      } else {
        test_assert_double_equal( 234.41583252  , enkf_plot_blockvector_iget( vector , 0 ));
        test_assert_double_equal( 234.95098877  , enkf_plot_blockvector_iget( vector , 1 ));
        test_assert_double_equal( 235.218841553 , enkf_plot_blockvector_iget( vector , 2 ));
        test_assert_double_equal( 235.486480713 , enkf_plot_blockvector_iget( vector , 3 ));
      }
    }

    enkf_plot_blockdata_free( block_data );
  }
}



int main( int argc , char ** argv) {
  const char * config_file = argv[1];
  util_install_signals();
  ert_test_context_type * test_context = ert_test_context_alloc("BLOCKDATA" , config_file , NULL );
  enkf_main_type * enkf_main = ert_test_context_get_main( test_context );

  test_assert_true( enkf_main_is_instance( enkf_main ));
  
  test_blockdata( enkf_main , "RFT2" , 50);
  test_blockdata( enkf_main , "RFT5" , 56);
  
  ert_test_context_free( test_context );
  exit(0);
}
