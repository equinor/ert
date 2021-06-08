/*
   Copyright (C) 2016  Equinor ASA, Norway.

   The file 'enkf_workflow_job_test_version.c' is part of ERT -
   Ensemble based Reservoir Tool.

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

#include <ert/util/test_util.h>
#include <ert/enkf/ert_test_context.hpp>




int main(int argc , const char ** argv) {
  enkf_main_install_SIGNALS();
  {
    const char * path = argv[1];
    ert_workflow_list_type * workflows = ert_workflow_list_alloc_empty(NULL);
    ert_workflow_list_add_jobs_in_directory( workflows , path );

    test_assert_true( ert_workflow_list_has_job( workflows , "CONF1"));
    test_assert_true( ert_workflow_list_has_job( workflows , "CONF2@1"));
    test_assert_false( ert_workflow_list_has_job( workflows , "CONF2"));
    test_assert_true( ert_workflow_list_has_job( workflows , "CONF3@1.2.3"));
    test_assert_true( ert_workflow_list_has_job( workflows , "CONF4@1.2.0"));
    test_assert_false( ert_workflow_list_has_job( workflows , "CONF5"));

    ert_workflow_list_free( workflows );
  }
  exit(0);
}
