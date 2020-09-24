/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'gen_kw_test.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/enkf/ert_test_context.hpp>

#include <ert/enkf/gen_kw.hpp>


void verify_parameters_txt( ) {
  int buffer_size = 0;
  char * file_content = util_fread_alloc_file_content("simulations/run0/parameters.txt", &buffer_size);

  stringlist_type * token_list = stringlist_alloc_from_split(file_content, " \n");
  double value = stringlist_iget_as_double(token_list, 5, NULL);

  test_assert_true(value > 0.0); //Verify precision
  test_assert_true(NULL != strstr(file_content, "LOG10_")); //Verify log entry

  stringlist_free(token_list);
  free(file_content);
}


void test_write_gen_kw_export_file(enkf_main_type * enkf_main) {
  enkf_fs_type * init_fs = enkf_main_get_fs( enkf_main );
  ensemble_config_type * ens_config = enkf_main_get_ensemble_config( enkf_main );
  enkf_node_type * enkf_node  = enkf_node_alloc( ensemble_config_get_node( ens_config , "MULTFLT" ));
  enkf_node_type * enkf_node2 = enkf_node_alloc( ensemble_config_get_node( ens_config , "MULTFLT2" ));
  test_assert_true(enkf_node_get_impl_type(enkf_node)  == GEN_KW);
  test_assert_true(enkf_node_get_impl_type(enkf_node2) == GEN_KW);

  {
    gen_kw_type * gen_kw  = (gen_kw_type *) enkf_node_value_ptr(enkf_node);
    gen_kw_type * gen_kw2 = (gen_kw_type *) enkf_node_value_ptr(enkf_node2);


    {
      rng_type * rng                       = rng_alloc( MZRAN , INIT_DEFAULT );
      const enkf_config_node_type * config = enkf_node_get_config(enkf_node);
      const int    data_size               = enkf_config_node_get_data_size( config, 0 );
      const double                    mean = 0.0; /* Mean and std are hardcoded - the variability should be in the transformation. */
      const double                    std  = 1.0;

      for (int i=0; i < data_size; ++i) {
        double random_number = enkf_util_rand_normal(mean , std , rng);
        gen_kw_data_iset(gen_kw,  i, random_number);
        gen_kw_data_iset(gen_kw2, i, random_number);
      }

      rng_free(rng);
    }
    node_id_type node_id = {.report_step = 0,
                            .iens        = 0 };

    enkf_node_store(enkf_node, init_fs, true, node_id);
    enkf_node_store(enkf_node2, init_fs, true, node_id);
  }

  {
    const subst_list_type * subst_list = subst_config_get_subst_list(enkf_main_get_subst_config(enkf_main));
    run_arg_type * run_arg = run_arg_alloc_INIT_ONLY( "run_id", init_fs , 0 ,0 , "simulations/run0", subst_list);
    enkf_state_ecl_write( enkf_main_get_ensemble_config( enkf_main ),
                          enkf_main_get_model_config( enkf_main ),
                          run_arg ,
                          init_fs);
    test_assert_true(util_file_exists("simulations/run0/parameters.txt"));
    run_arg_free( run_arg );
  }
  enkf_node_free( enkf_node );
  enkf_node_free( enkf_node2 );

  verify_parameters_txt( );
}



int main(int argc , char ** argv) {
  util_install_signals();
  {
    const char * config_file             =  argv[1];
    ert_test_context_type * test_context = ert_test_context_alloc("gen_kw_logarithmic_test" , config_file );
    enkf_main_type * enkf_main           = ert_test_context_get_main(test_context);

    test_assert_not_NULL(enkf_main);

    test_write_gen_kw_export_file(enkf_main);

    ert_test_context_free( test_context );
  }
  exit(0);
}

