/*
   Copyright (C) 2015  Statoil ASA, Norway.

   The file 'config_content.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/type_macros.h>

#include <ert/config/config_content.h>


#define CONFIG_CONTENT_TYPE_ID 6612520

struct config_content_struct {
  UTIL_TYPE_ID_DECLARATION;
  bool valid;
};


UTIL_IS_INSTANCE_FUNCTION( config_content , CONFIG_CONTENT_TYPE_ID )

config_content_type * config_content_alloc() {
  config_content_type * content = util_malloc( sizeof * content );
  UTIL_TYPE_ID_INIT( content , CONFIG_CONTENT_TYPE_ID );
  content->valid = false;
  return content;
}



void config_content_set_valid( config_content_type * content) {
  content->valid = true;
}

bool config_content_is_valid( const config_content_type * content ) {
  return content->valid;
}


void config_content_free( config_content_type * content ) {
  free( content );
}
