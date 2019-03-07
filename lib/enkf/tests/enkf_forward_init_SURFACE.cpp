/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_forward_init_SURFACE.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/util/test_work_area.hpp>
#include <ert/util/util.h>
#include <ert/res_util/arg_pack.hpp>
#include <ert/util/rng.h>
#include <ert/util/mzran.h>
#include <ert/res_util/subst_list.hpp>

#include <ert/enkf/enkf_main.hpp>
#include <ert/enkf/run_arg.hpp>

void create_runpath(enkf_main_type * enkf_main, int iter  ) {
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

void install_file( const ecl::util::TestArea& ta, const char * input_src_file ) {
  if (util_is_abs_path( input_src_file))
    return;
  else {
    std::string src_file = ta.original_path(input_src_file);
    char * src_path;

    util_alloc_file_components(input_src_file, &src_path, NULL, NULL);

    if (!util_entry_exists( src_path ))
      util_make_path( src_path );

    if (util_file_exists( src_file.c_str() )) {
      char * target_file   = util_alloc_filename( ta.test_cwd().c_str(), input_src_file, NULL );
      util_copy_file( src_file.c_str() , target_file );
      free( target_file );
    }

    free(src_path);
  }
}




int main(int argc , char ** argv) {
  enkf_main_install_SIGNALS();
  const char * root_path = argv[1];
  const char * config_file = argv[2];
  const char * init_file = argv[3];
  const char * forward_init_string = argv[4];

  ecl::util::TestArea ta("surface");
  ta.copy_directory_content(root_path);
  install_file(ta, init_file);
  {

    bool forward_init;
    bool strict = true;
    enkf_main_type * enkf_main;

    test_assert_true( util_sscanf_bool( forward_init_string , &forward_init));

    util_clear_directory( "Storage" , true , true );
    res_config_type * res_config = res_config_alloc_load(config_file);
    enkf_main = enkf_main_alloc(res_config, strict, true);
    ensemble_config_type * ens_config = enkf_main_get_ensemble_config( enkf_main );
    {
      const enkf_config_node_type * surface_config_node = ensemble_config_get_node( ens_config , "SURFACE");
      enkf_node_type * surface_node = enkf_node_alloc( surface_config_node );
      char * init_file1 = enkf_config_node_alloc_initfile( surface_config_node , NULL , 0);
      char * init_file2 = enkf_config_node_alloc_initfile( surface_config_node , "/tmp", 0);

      test_assert_bool_equal( enkf_config_node_use_forward_init( surface_config_node ) , forward_init );
      test_assert_string_equal( init_file1 , "Surface.irap");
      test_assert_string_equal( init_file2 , "/tmp/Surface.irap");

      free( init_file1 );
      free( init_file2 );

      rng_type * rng = rng_alloc( MZRAN, INIT_DEFAULT );
      test_assert_bool_equal( enkf_node_use_forward_init( surface_node ) , forward_init );
      if (forward_init)
        test_assert_bool_not_equal( enkf_node_initialize( surface_node , 0 , rng) , forward_init);
      // else hard_failure()
      enkf_node_free( surface_node );
      rng_free( rng );
    }
    test_assert_bool_equal( forward_init, ensemble_config_have_forward_init( enkf_main_get_ensemble_config( enkf_main )));

    if (forward_init) {
      const enkf_config_node_type * config_node = ensemble_config_get_node( ens_config , "SURFACE");
      enkf_state_type * state   = enkf_main_iget_state( enkf_main , 0 );
      enkf_fs_type * fs = enkf_main_get_fs( enkf_main );
      const subst_list_type * subst_list = subst_config_get_subst_list(enkf_main_get_subst_config(enkf_main));
      run_arg_type * run_arg = run_arg_alloc_ENSEMBLE_EXPERIMENT( "run_id", fs , 0 ,0 , "simulations/run0", "BASE", subst_list);
      enkf_node_type * surface_node = enkf_node_alloc( config_node );
      node_id_type node_id = {.report_step = 0 ,
                              .iens = 0 };

      create_runpath( enkf_main, 0 );
      test_assert_true( util_is_directory( "simulations/run0" ));

      {
        int error;
        stringlist_type * msg_list = stringlist_alloc_new();


        test_assert_false( enkf_node_has_data( surface_node , fs, node_id ));

        util_unlink_existing( "simulations/run0/Surface.irap" );

        test_assert_false( enkf_node_forward_init( surface_node , "simulations/run0" , 0 ));
        error = ensemble_config_forward_init( ens_config , run_arg );
        test_assert_true(LOAD_FAILURE & error);

        {
          enkf_fs_type * fs = enkf_main_get_fs(enkf_main);
          state_map_type * state_map = enkf_fs_get_state_map(fs);
          state_map_iset(state_map, 0, STATE_INITIALIZED);
        }
        error = enkf_state_load_from_forward_model(state, run_arg , msg_list);
        stringlist_free( msg_list );
        test_assert_true(LOAD_FAILURE & error);
      }


      util_copy_file( init_file , "simulations/run0/Surface.irap");
      {
        int error;
        stringlist_type * msg_list = stringlist_alloc_new();


        test_assert_true( enkf_node_forward_init( surface_node , "simulations/run0" , 0 ));
        error = ensemble_config_forward_init( ens_config , run_arg );
        test_assert_int_equal(0, error);
        error = enkf_state_load_from_forward_model( state , run_arg ,  msg_list );
        stringlist_free( msg_list );
        test_assert_int_equal(0, error);

        {
          double value;
          test_assert_true( enkf_node_user_get( surface_node , fs , "0" , node_id , &value));
          test_assert_double_equal( 2735.7461 , value);

          test_assert_true( enkf_node_user_get( surface_node , fs , "5" , node_id , &value));
          test_assert_double_equal( 2737.0122 , value);
        }
      }
      util_clear_directory( "simulations" , true , true );
      create_runpath( enkf_main, 0 );
      test_assert_true( util_is_directory( "simulations/run0" ));
      test_assert_true( util_is_file( "simulations/run0/SURFACE.INC" ));
      test_assert_true( enkf_node_fload( surface_node , "simulations/run0/SURFACE.INC"));
      {
        double value;
        test_assert_true( enkf_node_user_get( surface_node , fs , "0" , node_id , &value));
        test_assert_double_equal( 2735.7461 , value);

        test_assert_true( enkf_node_user_get( surface_node , fs , "5" , node_id , &value));
        test_assert_double_equal( 2737.0122 , value);
      }
      util_clear_directory( "simulations" , true , true );
    }
    enkf_main_free( enkf_main );
    res_config_free(res_config);
  }
}

