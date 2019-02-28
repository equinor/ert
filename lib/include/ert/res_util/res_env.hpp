/*
   Copyright (C) 2018  Equinor ASA, Norway.

   The file 'res_env.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef RESENV_H
#define RESENV_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

  char       ** res_env_alloc_PATH_list();
  char       *  res_env_alloc_PATH_executable(const char * executable );
  void          res_env_setenv( const char * variable , const char * value);
  const char *  res_env_interp_setenv( const char * variable , const char * value);
  void          res_env_unsetenv( const char * variable);
  char       *  res_env_alloc_envvar( const char * value );
  char       *  res_env_isscanf_alloc_envvar( const char * string , int env_index );
  const char *  res_env_update_path_var(const char * variable, const char * value, bool append);

#ifdef __cplusplus
}
#endif
#endif // RESLOG_H
