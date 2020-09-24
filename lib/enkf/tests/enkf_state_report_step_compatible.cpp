/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_state_report_step_compatible.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/util/util.h>

#include <ert/enkf/enkf_main.hpp>


bool check_ecl_sum_compatible(const enkf_main_type * enkf_main)
{
  stringlist_type * msg_list = stringlist_alloc_new();
  enkf_state_type * state    = enkf_main_iget_state( enkf_main , 0 );
  enkf_fs_type    * fs       = enkf_main_get_fs( enkf_main );
  char * job_name            = model_config_alloc_jobname( enkf_main_get_model_config( enkf_main ), 0);
  const subst_list_type * subst_list =  subst_config_get_subst_list(enkf_main_get_subst_config(enkf_main));

  run_arg_type * run_arg     = run_arg_alloc_ENSEMBLE_EXPERIMENT("run_id",
                                                                 fs,
                                                                 0,
                                                                 0,
                                                                 "simulations/run0",
                                                                 job_name,
                                                                 subst_list);

  state_map_type * state_map = enkf_fs_get_state_map(fs);
  state_map_iset(state_map, 0, STATE_INITIALIZED);

  int error = enkf_state_load_from_forward_model( state , run_arg , msg_list );

  free( job_name );
  stringlist_free( msg_list );
  return (REPORT_STEP_INCOMPATIBLE & error) ? false : true;
}



int main(int argc , char ** argv) {
  enkf_main_install_SIGNALS();
  const char * root_path      = argv[1];
  const char * config_file    = argv[2];
  const char * compatible_str = argv[3];
  bool check_compatible;

  test_assert_true( util_sscanf_bool( compatible_str , &check_compatible));

  ecl::util::TestArea ta("compatible");
  ta.copy_directory_content(root_path);

  bool strict = true;
  res_config_type * res_config = res_config_alloc_load(config_file);
  enkf_main_type * enkf_main = enkf_main_alloc(res_config, strict, true);

  test_assert_bool_equal(check_compatible , check_ecl_sum_compatible(enkf_main));

  enkf_main_free( enkf_main );
  res_config_free(res_config);
}
