/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'ext_param.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <string.h>
#include <stdbool.h>
#include <stdio.h>

#include <ert/util/util.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/ext_param_config.hpp>
#include <ert/enkf/ext_param.hpp>
#include <ert/enkf/value_export.hpp>

GET_DATA_SIZE_HEADER(ext_param);


struct ext_param_struct {
  int                         __type_id;
  const ext_param_config_type * config;
  double                      * data;
  int                           size;
};

/*****************************************************************/



void ext_param_free(ext_param_type *ext_param) {
  free(ext_param->data);
  free(ext_param);
}





ext_param_type * ext_param_alloc(const ext_param_config_type * config) {
  ext_param_type * ext_param = (ext_param_type *)util_malloc(sizeof *ext_param );
  ext_param->__type_id     = EXT_PARAM;
  ext_param->config        = config;
  ext_param->size          = ext_param_config_get_data_size( config );
  ext_param->data          = (double * ) util_calloc( ext_param->size , sizeof * ext_param->data );
  return ext_param;
}


bool ext_param_key_set( ext_param_type * param, const char * key, double value) {
  int index = ext_param_config_get_key_index( param->config, key);
  if (index < 0)
    return false;

  param->data[index] = value;
  return true;
}


double ext_param_key_get( const ext_param_type * param, const char * key) {
  int index = ext_param_config_get_key_index( param->config, key);
  if (index < 0)
    util_abort("%s: invalid key:%s \n",__func__ , key);

  return param->data[index];
}


bool ext_param_iset( ext_param_type * param, int index , double value) {
  if (index >= param->size)
    return false;

  if (index < 0)
    return false;

  param->data[index] = value;
  return true;
}


const char* ext_param_iget_key(const ext_param_type * param, int index) {
  return ext_param_config_iget_key( param->config , index );
}

double ext_param_iget(const ext_param_type * param, int index) {
  if (index >= param->size)
    util_abort("%s: invalid index:%d - range: [0,%d) \n",__func__ , index , param->size);

  if (index < 0)
    util_abort("%s: invalid index:%d - range: [0,%d) \n",__func__ , index , param->size);

  return param->data[index];
}


void ext_param_json_export(const ext_param_type * ext_param, const char * json_file) {
  FILE * stream = util_mkdir_fopen( json_file , "w");
  fprintf(stream, "{\n");
  for (int index=0; index < ext_param->size; index++) {
    fprintf(stream, "\"%s\" : %g", ext_param_config_iget_key( ext_param->config , index ), ext_param->data[index]);

    if (index < (ext_param->size - 1))
      fprintf(stream, ",\n");
    else
      fprintf(stream, "\n");

  }
  fprintf(stream, "}\n");
  fclose( stream );
}

void ext_param_ecl_write(const ext_param_type * ext_param , const char * run_path , const char * base_file , value_export_type * export_value) {
  char * target_file;

  if (run_path)
    target_file = util_alloc_filename( run_path , base_file  , NULL);
  else
    target_file = util_alloc_string_copy( base_file );

  ext_param_json_export( ext_param , target_file );
  free( target_file );
}

bool ext_param_write_to_buffer(const ext_param_type *ext_param , buffer_type * buffer,  int report_step) {
  buffer_fwrite_int( buffer , EXT_PARAM );
  buffer_fwrite(buffer , ext_param->data , sizeof *ext_param->data , ext_param->size);
  return true;
}


void ext_param_read_from_buffer(ext_param_type * ext_param , buffer_type * buffer, enkf_fs_type * fs, int report_step) {
  const int data_size = ext_param_config_get_data_size( ext_param->config );
  enkf_util_assert_buffer_type( buffer, EXT_PARAM );
  buffer_fread(buffer , ext_param->data , sizeof *ext_param->data , data_size);
}

int ext_param_get_size( const ext_param_type * ext_param ) {
  return ext_param->size;
}


bool ext_param_has_key( const ext_param_type * ext_param , const char * key) {
  return ext_param_config_has_key( ext_param->config , key );
}


/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/
UTIL_SAFE_CAST_FUNCTION(ext_param , EXT_PARAM)
UTIL_SAFE_CAST_FUNCTION_CONST(ext_param , EXT_PARAM)
VOID_ALLOC(ext_param)
VOID_FREE(ext_param)
VOID_ECL_WRITE(ext_param)
VOID_WRITE_TO_BUFFER(ext_param)
VOID_READ_FROM_BUFFER(ext_param)

