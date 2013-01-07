/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'config_include_test.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <test_util.h>
#include <util.h>

#include <config.h>
#include <config_schema_item.h>

void parse_test(config_type * config , const char * config_path , const char * config_file , bool abs_path) {
  char  * file = util_alloc_filename( config_path , config_file , NULL);
  
  config_clear( config );
  if (config_parse( config , file , "--" , "INCLUDE" , NULL , CONFIG_UNRECOGNIZED_IGNORE , true )) {
  } else {
    config_error_type * error = config_get_errors( config );
    config_error_fprintf( error , stdout );
    test_error_exit("Hmm - parsing %s failed \n", file );
  }
  free( file );  
}


int main(int argc , char ** argv) {
  const char * abs_path    = argv[1];
  const char * config_file = argv[2];
  char * cwd = util_alloc_cwd();
  printf("abs:%s  file:%s  cwd:%s \n",abs_path , config_file , cwd);
  char * rel_path = util_alloc_rel_path( cwd , abs_path );
  config_type * config = config_alloc();

  {
    config_schema_item_type * schema_item;
    
    schema_item = config_add_schema_item( config , "PATH0" , true );
    config_schema_item_set_argc_minmax( schema_item , 1 , 1 , 1 , (const config_item_types  [1]) { CONFIG_PATH });

    schema_item = config_add_schema_item( config , "PATH1" , true );
    config_schema_item_set_argc_minmax( schema_item , 1 , 1 , 1 , (const config_item_types  [1]) { CONFIG_PATH });

    schema_item = config_add_schema_item( config , "PATH2" , true );
    config_schema_item_set_argc_minmax( schema_item , 1 , 1 , 1 , (const config_item_types  [1]) { CONFIG_PATH });

    schema_item = config_add_schema_item( config , "PATH3" , true );
    config_schema_item_set_argc_minmax( schema_item , 1 , 1 , 1 , (const config_item_types  [1]) { CONFIG_PATH });

    schema_item = config_add_schema_item( config , "PATH4" , true );
    config_schema_item_set_argc_minmax( schema_item , 1 , 1 , 1 , (const config_item_types  [1]) { CONFIG_PATH });
  }

  parse_test( config , rel_path , config_file , false );
  parse_test( config , abs_path , config_file , true );

  config_free( config );
  exit(0);
}

