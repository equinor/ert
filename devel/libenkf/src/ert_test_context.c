/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
   The file 'ert_test_context.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ert/util/util.h>
#include <ert/util/type_macros.h>
#include <ert/util/test_work_area.h>

#include <ert/enkf/ert_test_context.h>
#include <ert/enkf/enkf_main.h>


#define ERT_TEST_CONTEXT_TYPE_ID 99671055
struct ert_test_context_struct {
  UTIL_TYPE_ID_DECLARATION;
  enkf_main_type * enkf_main;
  test_work_area_type * work_area;
};

UTIL_IS_INSTANCE_FUNCTION( ert_test_context , ERT_TEST_CONTEXT_TYPE_ID )


ert_test_context_type * ert_test_context_alloc( const char * test_name , const char * model_config , const char * site_config) {
   ert_test_context_type * test_context = util_malloc( sizeof * test_context );
   UTIL_TYPE_ID_INIT( test_context , ERT_TEST_CONTEXT_TYPE_ID );
   if (util_file_exists(model_config)) {
     test_context->work_area = test_work_area_alloc("ERT-TEST-CONTEXT");
     test_work_area_set_store( test_context->work_area , true );
     test_work_area_copy_parent_content(test_context->work_area , model_config );
     {
       char * config_file = util_split_alloc_filename( model_config );
       test_context->enkf_main = enkf_main_bootstrap( site_config , config_file , true , false );
       free( config_file );
     }
   } else {
     test_context->enkf_main = NULL;
     test_context->work_area = NULL;
   }
   return test_context;
}



enkf_main_type * ert_test_context_get_main( ert_test_context_type * test_context ) {
  return test_context->enkf_main;
}



void ert_test_context_free( ert_test_context_type * test_context ) {
  if (test_context->enkf_main)
    enkf_main_free( test_context->enkf_main );
  
  if (test_context->work_area)
    test_work_area_free( test_context->work_area );
  
  free( test_context );
}


bool ert_test_context_install_workflow_job( ert_test_context_type * test_context , const char * job_name , const char * job_file) {
  if (util_file_exists( job_file )) {
    enkf_main_type * enkf_main = ert_test_context_get_main( test_context );
    ert_workflow_list_type * workflow_list = enkf_main_get_workflow_list( enkf_main );
    ert_workflow_list_add_job( workflow_list , job_name , job_file );
    return ert_workflow_list_has_job( workflow_list , job_name );
  } else
    return false;
}
