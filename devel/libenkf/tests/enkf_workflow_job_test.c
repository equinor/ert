/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_workflow_job_test.c' is part of ERT - Ensemble based Reservoir Tool.
    
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
#include <ert/enkf/ert_test_context.h>

#include <ert/util/util.h>
#include <ert/util/string_util.h>

#include <ert/enkf/enkf_main.h>
#include <ert/enkf/enkf_main_jobs.h>


void test_create_case_job(const char * config_file, const char * job_file) {
  ert_test_context_type * test_context = ert_test_context_alloc("CreateCaseJob" , config_file , NULL);
  stringlist_type * args = stringlist_alloc_new();
  stringlist_append_copy( args , "newly_created_case");
  ert_test_context_install_workflow_job( test_context , "JOB" , job_file );
  test_assert_true( ert_test_context_run_worklow_job( test_context , "JOB" , args) );

  char * new_case = util_alloc_filename( "storage" , "newly_created_case" , NULL);
  test_assert_true(util_is_directory(new_case));
  free(new_case);

  stringlist_free( args );
  ert_test_context_free( test_context );
}

void test_load_results_job(const char * config_file, const char * job_file) {
  ert_test_context_type * test_context = ert_test_context_alloc("LoadResultsJob" , config_file , NULL);
  stringlist_type * args = stringlist_alloc_new();
  ert_test_context_install_workflow_job( test_context , "JOB" , job_file );
  stringlist_append_copy( args , "0,1");
  test_assert_true( ert_test_context_run_worklow_job( test_context , "JOB" , args) );
  stringlist_free( args );
  ert_test_context_free( test_context );
}


void test_init_case(const char * config_file, const char * job_file) {
  ert_test_context_type * test_context = ert_test_context_alloc("InitCaseJob" , config_file , NULL);
  stringlist_type * args = stringlist_alloc_new();
  enkf_main_type * enkf_main = ert_test_context_get_main(test_context);

  ert_test_context_install_workflow_job( test_context , "JOB" , job_file );

  //Test init current case from existing
  enkf_fs_type * cur_fs = enkf_main_mount_alt_fs( enkf_main , "new_current_case" , false , true );
  enkf_main_select_fs(enkf_main, "new_current_case");

  stringlist_append_copy( args, "default"); //case to init from
  test_assert_true( ert_test_context_run_worklow_job( test_context , "JOB" , args) );

  enkf_fs_umount(cur_fs);
  stringlist_clear(args);

  //Test init case from existing case:
  stringlist_append_copy(args, "default"); //case to init from
  stringlist_append_copy(args, "new_not_current_case");
  test_assert_true( ert_test_context_run_worklow_job( test_context , "JOB" , args) );

  stringlist_free( args );
  ert_test_context_free( test_context );
}




int main(int argc , const char ** argv) {
  enkf_main_install_SIGNALS();
  
  const char * config_file               = argv[1];
  const char * job_file_create_case      = argv[2];
  const char * job_file_load_results     = argv[3];
  const char * job_file_init_case_job    = argv[4];

  test_create_case_job(config_file, job_file_create_case);
  test_load_results_job(config_file, job_file_load_results);
  test_init_case(config_file, job_file_init_case_job);

  exit(0);
}
