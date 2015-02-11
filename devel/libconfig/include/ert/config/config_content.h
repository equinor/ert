/*
   Copyright (C) 2015  Statoil ASA, Norway.

   The file 'config_content.h' is part of ERT - Ensemble based Reservoir Tool.

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


#ifndef __CONFIG_CONTENT_H__
#define __CONFIG_CONTENT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/type_macros.h>

typedef struct config_content_struct config_content_type;


  config_content_type * config_content_alloc();
  void config_content_free( config_content_type * content );
  void config_content_set_valid( config_content_type * content);
  bool config_content_is_valid( const config_content_type * content );

  UTIL_IS_INSTANCE_HEADER( config_content );

#ifdef __cplusplus
}
#endif
#endif
