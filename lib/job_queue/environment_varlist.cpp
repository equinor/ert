/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'environment_varlist.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/job_queue/environment_varlist.hpp>

#include <ert/res_util/res_env.hpp>

#include <ert/util/hash.hpp>

#define ENV_VAR_KEY_STRING     "global_environment"
#define UPDATE_PATH_KEY_STRING "global_update_path"

struct env_varlist_struct {
  hash_type * varlist;
  hash_type * updatelist;
};

env_varlist_type * env_varlist_alloc() {
  env_varlist_type * list = (env_varlist_type*)util_malloc( sizeof * list );
  list->varlist = hash_alloc();
  list->updatelist = hash_alloc();
  return list;
}

void env_varlist_update_path(env_varlist_type * list, const char * path_var, const char * new_path)  {
  hash_insert_string( list->updatelist, path_var , res_env_update_path_var( path_var , new_path , false));
}

void env_varlist_setenv(env_varlist_type * list, const char * key, const char * value) {
  const char * interp_value = res_env_interp_setenv(key, value);
  hash_insert_string(list->varlist, key, interp_value);
}


static void env_varlist_fprintf_hash(const hash_type * list, const char * keystring, FILE * stream) {
  int size = hash_get_size(list);
  fprintf(stream, "\"%s\" : {", keystring);
  stringlist_type * stringlist = hash_alloc_stringlist(list);
  int i_max = size - 1;
  for (int i = 0; i < size; i++) {
    const char * key = stringlist_iget(stringlist, i);
    fprintf(stream, "\"%s\" : \"%s\"", key, (char*)hash_get(list, key)   );
    if (i < i_max)
      fprintf(stream, ", ");
  }
  fprintf(stream, "}");
  stringlist_free(stringlist);
}

void env_varlist_json_fprintf(const env_varlist_type * list, FILE * stream) {
  env_varlist_fprintf_hash(list->varlist,    ENV_VAR_KEY_STRING,     stream); fprintf(stream, ",\n");
  env_varlist_fprintf_hash(list->updatelist, UPDATE_PATH_KEY_STRING, stream);
}

int env_varlist_get_size(env_varlist_type * list) {
  return hash_get_size(list->varlist);
}

void env_varlist_free(env_varlist_type * list) {
  hash_free(list->varlist);
  hash_free(list->updatelist);
  free(list);
}
