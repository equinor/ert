/*
   Copyright (C) 2014  Statoil ASA, Norway.

   The file 'enkf_plot_gen_kw_fs.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/enkf/enkf_config_node.h>
#include <ert/enkf/enkf_plot_gen_kw.h>
#include <ert/enkf/ert_test_context.h>




void test_gen_kw( enkf_main_type * enkf_main , const char * gen_kw_key , const char * param_key , int report_step ) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config( enkf_main );
  test_assert_true( ensemble_config_has_key( ensemble_config , gen_kw_key ) );
  const enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , gen_kw_key );

  {
    enkf_fs_type * fs = enkf_main_get_fs( enkf_main );

    enkf_plot_gen_kw_type * gen_kw_data = enkf_plot_gen_kw_alloc( config_node );

    test_assert_true( enkf_plot_gen_kw_is_instance( gen_kw_data ) );
    enkf_plot_gen_kw_load( gen_kw_data , fs , report_step , FORECAST , param_key, NULL );
    test_assert_int_equal( enkf_main_get_ensemble_size( enkf_main ) , enkf_plot_gen_kw_get_size( gen_kw_data ));

    {

      /* if (report_step == 0) { */
      test_assert_double_equal( -0.6975 , enkf_plot_gen_kw_iget( gen_kw_data ,  0 ));
      test_assert_double_equal( -0.299  , enkf_plot_gen_kw_iget( gen_kw_data ,  1 ));
      test_assert_double_equal( -0.7383 , enkf_plot_gen_kw_iget( gen_kw_data , 99 ));
    }

    enkf_plot_gen_kw_free( gen_kw_data );
  }
}



int main( int argc , char ** argv) {
  const char * config_file = argv[1];
  util_install_signals();
  ert_test_context_type * test_context = ert_test_context_alloc("GEN_KW_DATA" , config_file , NULL );
  enkf_main_type * enkf_main = ert_test_context_get_main( test_context );

  test_gen_kw( enkf_main , "FAULT_TRANSLATE_REL" , "TRANS_FAC", 0);

  ert_test_context_free( test_context );
  exit(0);
}
