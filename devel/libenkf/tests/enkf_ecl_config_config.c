/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_ecl_config_config.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <ert/util/path_stack.h>

#include <ert/config/config.h>

#include <ert/enkf/ecl_config.h>
#include <ert/enkf/ecl_refcase_list.h>

int main(int argc , char ** argv) {
  char * config_path;
  char * config_file;
  path_stack_type * path_stack = path_stack_alloc( );
  path_stack_push_cwd( path_stack );
  {
    util_alloc_file_components( argv[1] , &config_path , &config_file , NULL);
    printf("Changing to: %s \n",config_path);
    printf("Using : %s \n",config_file);
    if (config_path)
      chdir( config_path );
    
    {
      config_type * config = config_alloc();
      ecl_config_type * ecl_config = ecl_config_alloc_empty();
      ecl_refcase_list_type * refcase_list = ecl_config_get_refcase_list( ecl_config );
      
      ecl_config_add_config_items( config );
      test_assert_true( config_parse( config , config_file , "--" , NULL , NULL , CONFIG_UNRECOGNIZED_WARN , true));
      ecl_config_init( ecl_config , config );
      test_assert_true( ecl_config_has_refcase( ecl_config ));
      test_assert_int_equal( ecl_refcase_list_get_size( refcase_list) , 17);

      ecl_config_free( ecl_config );
    }
  }
  path_stack_pop( path_stack );
  exit(0);
}

