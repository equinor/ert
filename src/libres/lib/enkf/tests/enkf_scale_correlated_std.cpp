/*
   Copyright (C) 2015  Equinor ASA, Norway.

   The file 'enkf_scale_correlated_std.c' is part of ERT - Ensemble based Reservoir Tool.

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


int main(int argc , const char ** argv) {
  const char * config_file = argv[1];
  const char * workflow_job_file = argv[2];
  enkf_main_install_SIGNALS();
  {
    ert_test_context_type * test_context = ert_test_context_alloc("std_scale_test" , config_file);
    stringlist_type * args = stringlist_alloc_new();

    ert_test_context_install_workflow_job( test_context , "STD_SCALE" , workflow_job_file );
    stringlist_append_copy( args, "WWCT:OP_1");
    test_assert_true( ert_test_context_run_worklow_job(test_context, "STD_SCALE", args));

    stringlist_append_copy( args, "WWCT:OP_2");
    test_assert_true( ert_test_context_run_worklow_job(test_context, "STD_SCALE", args));

    stringlist_clear(args);
    stringlist_append_copy( args, "RPR2_1");
    stringlist_append_copy( args, "RPR2_2");
    stringlist_append_copy( args, "RPR2_3");
    stringlist_append_copy( args, "RPR2_4");
    stringlist_append_copy( args, "RPR2_5");
    stringlist_append_copy( args, "RPR2_6");
    stringlist_append_copy( args, "RPR2_7");
    stringlist_append_copy( args, "RPR2_8");
    test_assert_true( ert_test_context_run_worklow_job(test_context, "STD_SCALE", args));

    stringlist_free( args );
    ert_test_context_free( test_context );
  }
  exit(0);
}
