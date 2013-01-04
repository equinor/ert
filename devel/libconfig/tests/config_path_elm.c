/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'config_path_elm.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <config.h>
#include <config_path_elm.h>



int main(int argc , char ** argv) {
#ifdef ERT_LINUX
  const char * root = "/tmp/root";
  const char * rel_path = "rel/path";
  const char * abs_path = "/tmp/root/abs/path";
  const char * rel_true = "rel/path/XXX";
  const char * abs_true = "/tmp/root/rel/path/XXX";
#endif

  {
    config_path_elm_type * path_elm = config_path_elm_alloc( root , rel_path );
  
    test_string_equal( config_path_elm_get_path( path_elm ) , rel_path );
    test_string_equal( config_path_elm_get_abspath( path_elm ) , abs_path );
    test_string_equal( config_path_elm_alloc_path( path_elm , "XXX" ) , rel_true);
    test_string_equal( config_path_elm_alloc_abspath( path_elm , "XXX" ) , abs_true);


    config_path_elm_free( path_elm );
  }
  printf("test1 OK \n");
  {
    config_path_elm_type * path_elm = config_path_elm_alloc( root , abs_path );
  
    test_string_equal( config_path_elm_get_path( path_elm ) , rel_path );
    test_string_equal( config_path_elm_get_abspath( path_elm ) , abs_path );
    test_string_equal( config_path_elm_alloc_path( path_elm , "XXX" ) , rel_true);
    test_string_equal( config_path_elm_alloc_abspath( path_elm , "XXX" ) , abs_true);

    config_path_elm_free( path_elm );
  }
  printf("test2 OK \n");
  exit(0);
}

