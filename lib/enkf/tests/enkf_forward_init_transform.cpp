/*
   Copyright (C) 2013  Statoil ASA, Norway.

   The file 'enkf_forward_init_transform.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>
#include <ert/util/util.h>
#include <ert/res_util/subst_list.hpp>

#include <ert/ecl/ecl_kw_magic.h>

#include <ert/enkf/enkf_main.hpp>
#include <ert/enkf/run_arg.hpp>
#include <ert/enkf/enkf_config_node.hpp>






void create_runpath(enkf_main_type * enkf_main, int iter ) {
  const int ens_size         = enkf_main_get_ensemble_size( enkf_main );
  bool_vector_type * iactive = bool_vector_alloc(ens_size, true);
  const path_fmt_type * runpath_fmt = model_config_get_runpath_fmt( enkf_main_get_model_config( enkf_main ));
  const subst_list_type * subst_list = subst_config_get_subst_list(enkf_main_get_subst_config(enkf_main));
  enkf_fs_type * fs           =  enkf_main_get_fs(enkf_main);
  ert_run_context_type * run_context = ert_run_context_alloc_INIT_ONLY( fs, INIT_CONDITIONAL, iactive, runpath_fmt, subst_list , iter );

  enkf_main_create_run_path(enkf_main , run_context);
  bool_vector_free(iactive);
  ert_run_context_free( run_context );
}


bool check_original_exported_data_equal(const enkf_node_type * field_node) {
  FILE * original_stream = util_fopen( "petro.grdecl" , "r");
  ecl_kw_type * kw_original = ecl_kw_fscanf_alloc_grdecl_dynamic( original_stream , "PORO" , ECL_DOUBLE );

  enkf_node_ecl_write(field_node, "tmp", NULL, 0);
  FILE * exported_stream = util_fopen( "tmp/PORO.grdecl" , "r");
  ecl_kw_type * kw_exported = ecl_kw_fscanf_alloc_grdecl_dynamic( exported_stream , "PORO" , ECL_DOUBLE );

  bool ret = ecl_kw_numeric_equal(kw_original, kw_exported, 1e-5 , 1e-5);

  fclose(original_stream);
  fclose(exported_stream);
  ecl_kw_free(kw_original);
  ecl_kw_free(kw_exported);

  return ret;
}


int main(int argc , char ** argv) {
  enkf_main_install_SIGNALS();
  const char * root_path   = argv[1];
  const char * config_file = argv[2];
  const char * init_file   = argv[3];
  const char * forward_init_string = argv[4];

  test_work_area_type * work_area = test_work_area_alloc__(config_file, true);
  test_work_area_copy_directory_content( work_area , root_path );
  test_work_area_install_file( work_area , init_file );

  bool strict = true;
  res_config_type * res_config = res_config_alloc_load(config_file);
  enkf_main_type * enkf_main = enkf_main_alloc(res_config, strict, true);
  ensemble_config_type * ens_config = enkf_main_get_ensemble_config( enkf_main );
  enkf_fs_type * init_fs = enkf_main_get_fs(enkf_main);
  const subst_list_type * subst_list = subst_config_get_subst_list(enkf_main_get_subst_config(enkf_main));
  run_arg_type * run_arg = run_arg_alloc_ENSEMBLE_EXPERIMENT( "run_id", init_fs , 0 ,0 , "simulations/run0", "base", subst_list);
  enkf_config_node_type * config_node = ensemble_config_get_node( ens_config , "PORO");
  enkf_node_type * field_node = enkf_node_alloc( config_node );

  bool forward_init;
  test_assert_true( util_sscanf_bool( forward_init_string , &forward_init));
  test_assert_bool_equal( enkf_node_use_forward_init( field_node ) , forward_init );

  util_clear_directory( "Storage" , true , true );

  create_runpath( enkf_main, 0 );
  test_assert_true( util_is_directory( "simulations/run0" ));

  if (forward_init)
    util_copy_file( init_file , "simulations/run0/petro.grdecl");

  {
    enkf_state_type * state   = enkf_main_iget_state( enkf_main , 0 );
    bool_vector_type * iactive = bool_vector_alloc( enkf_main_get_ensemble_size(enkf_main) , true);
    int error;
    stringlist_type * msg_list = stringlist_alloc_new();
    error = enkf_state_load_from_forward_model( state , run_arg ,  msg_list );
    stringlist_free( msg_list );
    bool_vector_free( iactive );
    test_assert_int_equal(error, 0);
  }

  test_assert_true(check_original_exported_data_equal(field_node));

  run_arg_free( run_arg );
  enkf_main_free(enkf_main);
  res_config_free(res_config);
  test_work_area_free(work_area);
}

