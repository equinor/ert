/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_state_manual_load_test.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/util/test_work_area.hpp>

#include <ert/enkf/enkf_main.hpp>


int test_load_manually_to_new_case(enkf_main_type * enkf_main) {
  int result = 0;
  int iens = 0;
  int iter = 0;
  const char * casename = "new_case";
  char * job_name  = model_config_alloc_jobname( enkf_main_get_model_config( enkf_main ), 0);
  enkf_main_select_fs( enkf_main , casename );


  enkf_fs_type * fs = enkf_main_get_fs( enkf_main );
  const subst_list_type * subst_list = subst_config_get_subst_list(enkf_main_get_subst_config(enkf_main));
  run_arg_type * run_arg = run_arg_alloc_ENSEMBLE_EXPERIMENT("run_id",
                                                             fs,
                                                             iens,
                                                             iter,
                                                             "simulations/run0",
                                                             job_name,
                                                             subst_list);
  {
    arg_pack_type * arg_pack = arg_pack_alloc();
    arg_pack_append_ptr( arg_pack , enkf_main_iget_state(enkf_main, 0));
    arg_pack_append_ptr( arg_pack , run_arg );
    arg_pack_append_owned_ptr( arg_pack , stringlist_alloc_new() , stringlist_free__);
    arg_pack_append_bool( arg_pack, true );
    arg_pack_append_ptr( arg_pack, &result );

    enkf_state_load_from_forward_model_mt(arg_pack);
    arg_pack_free(arg_pack);
  }
  free(job_name);
  return result;
}




int main(int argc , char ** argv) {
  enkf_main_install_SIGNALS();
  const char * root_path   = argv[1];
  const char * config_file = argv[2];

  ecl::util::TestArea ta(config_file);
  ta.copy_directory_content(root_path);
  {
    bool strict = true;
    res_config_type * res_config = res_config_alloc_load(config_file);
    enkf_main_type * enkf_main = enkf_main_alloc(res_config, strict, true);

    test_assert_int_equal( 0 , test_load_manually_to_new_case(enkf_main));

    enkf_main_free( enkf_main );
    res_config_free(res_config);
  }
  exit(0);
}

