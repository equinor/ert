/*
   Copyright (C) 2017  Statoil ASA, Norway.

   The file 'enkf_workflow_job_test2.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/enkf/ert_test_context.hpp>

#include <ert/util/util.h>
#include <ert/util/string_util.h>

#include <ert/enkf/enkf_main.hpp>
#include <ert/enkf/enkf_main_jobs.hpp>


ert_test_context_type * create_context( const char * config_file, const char * name ) {
  ert_test_context_type * test_context = ert_test_context_alloc__(name , config_file, true);
  usleep( 2 * 100000 );
  return test_context;
}



void test_load_results_job(ert_test_context_type * test_context , const char * job_name , const char * job_file) {
  stringlist_type * args = stringlist_alloc_new();
  ert_test_context_install_workflow_job( test_context , job_name , job_file );
  stringlist_append_copy( args , "0");
  stringlist_append_copy( args , ",");
  stringlist_append_copy( args , "1");
  test_assert_true( ert_test_context_run_worklow_job( test_context , job_name , args) );
  stringlist_free( args );

  enkf_main_type * enkf_main = ert_test_context_get_main( test_context );
  enkf_fs_type * fs = enkf_main_get_fs( enkf_main );
  time_map_type * time_map = enkf_fs_get_time_map( fs );
  test_assert_true( time_map_get_last_step( time_map ) > 0 );
}



int main(int argc , const char ** argv) {
  enkf_main_install_SIGNALS();

  const char * config_file                  = argv[1];
  const char * job_file_load_results        = argv[2];

  ert_test_context_type * test_context = create_context( config_file, "enkf_workflow_job_test2" );
  test_load_results_job(test_context, "JOB" , job_file_load_results);
  ert_test_context_free( test_context );

  exit(0);
}
