/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_export_field_test.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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


void test_int_vector(const int_vector_type * list , int length , ...) {
  va_list ap;
  int i;
  va_start(ap , length);
  test_assert_int_equal( length , int_vector_size( list ));

  for (i =0; i < int_vector_size( list ); i++) {
    int value = va_arg(ap , int);
    test_assert_int_equal( int_vector_iget( list , i ) , value);
  }

  va_end(ap);
}


int main(int argc , char ** argv) {
  enkf_main_install_SIGNALS();
  
  const char * config_path = argv[1];
  const char * config_file = argv[2];
  const char * field       = argv[3];
  const char * file_name   = argv[4];
  bool check_range  = (0 == strcmp(argv[5], "TRUE")) ? true : false; 
   
  test_work_area_type * work_area = test_work_area_alloc(config_file );
  test_work_area_copy_directory_content( work_area , config_path );
  test_work_area_set_store(work_area, true); 
  
  int_vector_type * realization_list = string_util_alloc_active_list(""); //Realizations range: rest of input arguments
  int arg_index = 6;
  for (; arg_index < argc; ++arg_index) {  
    string_util_update_active_list(argv[arg_index], realization_list); 
  }
  
  if (check_range)
    test_int_vector( realization_list , 3 , 0,1,2);

  if (util_char_in('%', strlen(file_name), file_name) && util_char_in('d', strlen(file_name), file_name)) {
    enkf_main_type * enkf_main = enkf_main_bootstrap( NULL , config_file , true , true );
    const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
    field_file_format_type file_type = field_config_default_export_format(file_name); 
    enkf_main_export_field(enkf_main, ensemble_config, field, file_name, realization_list, file_type);
    enkf_main_free( enkf_main );
  } else
      printf("There must be a %%d in the file name\n"); 
  
  test_work_area_free(work_area); 
  int_vector_free(realization_list);  
}
