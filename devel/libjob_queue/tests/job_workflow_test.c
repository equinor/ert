/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'job_workflow_test.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>

#include <test_util.h>
#include <util.h>

#include <config.h>
#include <workflow.h>
#include <workflow_job.h>
#include <workflow_joblist.h>


void create_workflow( const char * workflow_file , const char * tmp_file , int value) {
  FILE * stream  = util_fopen( workflow_file , "w");
  fprintf(stream , "CREATE_FILE   %s   %d\n" , tmp_file , value);
  fprintf(stream , "READ_FILE     %s\n" , tmp_file );
  fclose( stream );
  
  printf("Have created:%s \n",workflow_file );
}


void read_file( void * self , const stringlist_type * args) {
  printf("Running read_file \n");
  int * value = (int *) self;
  FILE * stream = util_fopen(stringlist_iget(args , 0 ) , "r");
  fscanf(stream , "%d" , value );
  fclose( stream );
}



int main( int argc , char ** argv) {
  int int_value = rand();
  int read_value = 100;
  workflow_joblist_type * joblist = workflow_joblist_alloc();

  if (!workflow_joblist_add_job_from_file( joblist , "CREATE_FILE" , argv[1])) 
    test_error_exit("Loading job CREATE_FILE failed\n");

  if (!workflow_joblist_add_job_from_file( joblist , "READ_FILE"   , argv[2]))
    test_error_exit("Loading job READ_FILE failed\n");

  {
    config_type * workflow_compiler = workflow_joblist_get_compiler( joblist );
    
    if (config_get_schema_size( workflow_compiler ) != 2)
      test_error_exit("Config compiler - wrong size \n");
  }
  

  {
    const char * workflow_file = "/tmp/workflow";
    const char * tmp_file = "/tmp/fileX";
    workflow_type * workflow;
    
    create_workflow( workflow_file , tmp_file , int_value );
    workflow = workflow_alloc(workflow_file , joblist );
    unlink( workflow_file );
    
    if (!workflow_run( workflow , &read_value )) {
      config_type * workflow_compiler = workflow_joblist_get_compiler( joblist );
      config_fprintf_errors( workflow_compiler , stdout);
      unlink( tmp_file );
      test_error_exit("Workflow did not run\n");
    }
    unlink( tmp_file );
  }
  workflow_joblist_free( joblist );
  if (int_value != read_value)
    test_error_exit("Wrong numeric value read back \n");
  exit(0);
}
