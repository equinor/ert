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
#include <ert/enkf/enkf_plot_gendata.h>




void test_gendata( enkf_main_type * enkf_main , const char * obs_key , int report_step ) {
  enkf_obs_type * enkf_obs = enkf_main_get_obs( enkf_main );
  obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key);
  {
    enkf_plot_gendata_type * gen_data = enkf_plot_gendata_alloc_from_obs_vector( obs_vector );
    enkf_fs_type * fs = enkf_main_get_fs( enkf_main );

    enkf_plot_gendata_load(gen_data, fs, report_step, FORECAST, NULL);

    test_assert_int_equal( enkf_main_get_ensemble_size( enkf_main ) , enkf_plot_gendata_get_size( gen_data ));
    {
      enkf_plot_genvector_type * vector = enkf_plot_gendata_iget( gen_data , 0);
      test_assert_true( enkf_plot_genvector_is_instance( vector ));

      /*
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
      */
    }

    {
      enkf_plot_genvector_type * vector = enkf_plot_gendata_iget( gen_data , 9 );
      test_assert_true( enkf_plot_genvector_is_instance( vector ));
      /*
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
      */
    }

    enkf_plot_gendata_free( gen_data );
  }

}



int main( int argc , char ** argv) {
  const char * config_file = argv[1];
  util_install_signals();
  ert_test_context_type * test_context = ert_test_context_alloc("GENDATA" , config_file , NULL );
  enkf_main_type * enkf_main = ert_test_context_get_main( test_context );
  
  test_gendata( enkf_main , "RFT2" , 60);
  test_gendata( enkf_main , "RFT5" , 61);
  
  ert_test_context_free( test_context );
  exit(0);
}
