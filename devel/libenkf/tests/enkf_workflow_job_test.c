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
#include <ert/util/util.h>
#include <ert/util/string_util.h>

#include <ert/enkf/enkf_main.h>
#include <ert/enkf/enkf_main_jobs.h>

void test_smoother_job(ert_workflow_list_type * workflow_list) {
   test_assert_true(ert_workflow_list_has_job(workflow_list, "RUN_SMOOTHER"));
}

void test_create_case_job(enkf_main_type * enkf_main, ert_workflow_list_type * workflow_list) {
    test_assert_true(ert_workflow_list_has_job(workflow_list, "CREATE_CASE"));

    char * new_case = util_alloc_filename( "storage" , "created_case" , NULL);
    test_assert_false(util_file_exists(new_case));
    stringlist_type * args = stringlist_alloc_new();
    stringlist_append_copy( args , "created_case");
    enkf_main_create_case_JOB( enkf_main , args );
    test_assert_true(util_file_exists(new_case));
    free(new_case);
    stringlist_free(args);
}

void test_create_case_workflow(enkf_main_type * enkf_main, ert_workflow_list_type * workflow_list) {
    ert_workflow_list_run_workflow(workflow_list  , "CREATE_CASE" , enkf_main);
    test_assert_true(util_file_exists("storage/my_new_case"));
}

int main(int argc , const char ** argv) {
  enkf_main_install_SIGNALS();
  
  const char * config_path  = argv[1];
  const char * config_file  = argv[2];
  const char * job_dir_path = argv[3]; 
    
  test_work_area_type * work_area = test_work_area_alloc(config_file );
  test_work_area_set_store(work_area, true); 
  test_work_area_copy_directory_content( work_area , config_path );

  enkf_main_type * enkf_main = enkf_main_bootstrap( NULL , config_file , true , true );  
  run_mode_type run_mode = ENSEMBLE_EXPERIMENT; 
  bool_vector_type * iactive = bool_vector_alloc(3, true); 
  enkf_main_init_run(enkf_main , iactive, run_mode , INIT_FORCE);     /* This is ugly */
  bool_vector_free(iactive); 

  ert_workflow_list_type * workflow_list = enkf_main_get_workflow_list(enkf_main);
  log_type * logh = enkf_main_get_logh(enkf_main);
  ert_workflow_list_add_jobs_in_directory(workflow_list, job_dir_path, logh);

  test_smoother_job(workflow_list);
  test_create_case_job(enkf_main, workflow_list);
  test_create_case_workflow(enkf_main, workflow_list);


  enkf_main_free( enkf_main );
  test_work_area_free(work_area); 
  exit(0);
}
