/*
   Copyright (C) 2016  Equinor ASA, Norway.

   This file is part of ERT - Ensemble based Reservoir Tool.

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
#include <string.h>

#include <ert/enkf/ert_test_context.hpp>
#include <ert/util/test_util.h>


int main(int argc , char ** argv) {
  enkf_main_install_SIGNALS();

  const char * config_file             = argv[1];
  ert_test_context_type * test_context = ert_test_context_alloc("VerifyJobsFileTest" , config_file);
  enkf_main_type * enkf_main           = ert_test_context_get_main(test_context);

  {
    const int ens_size         = enkf_main_get_ensemble_size( enkf_main );
    bool_vector_type * iactive = bool_vector_alloc(ens_size, true);
    const path_fmt_type * runpath_fmt = model_config_get_runpath_fmt( enkf_main_get_model_config( enkf_main ));
    const subst_list_type * subst_list = NULL;
    enkf_fs_type * fs                  =  enkf_main_get_fs(enkf_main);
    ert_run_context_type * run_context = ert_run_context_alloc_INIT_ONLY( fs , INIT_CONDITIONAL, iactive, runpath_fmt, subst_list , 0 );

    enkf_main_create_run_path(enkf_main , run_context );
    ert_run_context_free( run_context );
    bool_vector_free(iactive);
  }

  const char * filename = util_alloc_filename(ert_test_context_get_cwd(test_context),
                                              "simulations/run0/jobs.json", NULL);
  const char * jobs_file_content = util_fread_alloc_file_content(filename, NULL);

  test_assert_true  (strstr(jobs_file_content, "\"umask\" : \"0022\"") != NULL);
  test_assert_false (strstr(jobs_file_content, "\"umask\" : \"0032\"") != NULL);
  test_assert_false (strstr(jobs_file_content, "\"umask\" : \"0122\"") != NULL);
  test_assert_false (strstr(jobs_file_content, "\"umask\" : \"1022\"") != NULL);

  ert_test_context_free(test_context);
  exit(0);
}
