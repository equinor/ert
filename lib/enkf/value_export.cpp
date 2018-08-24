/*
   Copyright (C) 2017  Statoil ASA, Norway.

   The file 'value_export.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <math.h>

#include <ert/util/stringlist.h>
#include <ert/util/double_vector.h>

#include <ert/enkf/value_export.hpp>

#define VALUE_EXPORT_TYPE_ID     5741761


struct value_export_struct {
  UTIL_TYPE_ID_DECLARATION;
  char * directory;
  char * base_name;

  stringlist_type * keys;
  double_vector_type * values;
};


static void backup_if_existing(const char * filename) {
  if(util_file_exists(filename)) {
    char * backup_file_name = util_alloc_filename(NULL, filename, "old");
    util_move_file(filename, backup_file_name);
    free(backup_file_name);
  }
}


value_export_type * value_export_alloc(const char * directory, const char * base_name) {
  value_export_type * value = (value_export_type *)util_malloc( sizeof * value );
  UTIL_TYPE_ID_INIT( value , VALUE_EXPORT_TYPE_ID );
  value->directory = util_alloc_string_copy( directory );
  value->base_name = util_alloc_string_copy( base_name );

  value->keys = stringlist_alloc_new( );
  value->values = double_vector_alloc(0,0);
  return value;
}


void value_export_free(value_export_type * value) {
  stringlist_free( value->keys );
  double_vector_free( value->values );
  free( value->directory );
  free( value->base_name );
  free( value );
}

int value_export_size( const value_export_type * value) {
  return double_vector_size( value->values );
}


void value_export_txt__(const value_export_type * value, const char * filename) {
  const int size = double_vector_size( value->values );
  if (size > 0) {
    FILE * stream = util_fopen( filename , "w");

    for (int i=0; i < size; i++) {
      const char * key          = stringlist_iget( value->keys, i );
      double double_value              = double_vector_iget( value->values, i );
      fprintf(stream, "%s %g\n", key, double_value);
    }
    fclose( stream );
  }
}

void value_export_txt(const value_export_type * value) {
  char * filename = util_alloc_filename( value->directory , value->base_name, "txt");
  backup_if_existing(filename);
  value_export_txt__( value, filename );
  free( filename );
}

void value_export_json(const value_export_type * value) {
  char * filename = util_alloc_filename( value->directory , value->base_name, "json");
  backup_if_existing(filename);
  const int size = double_vector_size( value->values );
  if (size > 0) {
    FILE * stream = util_fopen( filename , "w");
    fprintf(stream, "{\n");
    for (int i=0; i < size; i++) {
      const char * key          = stringlist_iget( value->keys, i );
      double double_value       = double_vector_iget( value->values, i );
      if (isnan(double_value))
        fprintf(stream,"\"%s\" : NaN", key);
      else
        fprintf(stream, "\"%s\" : %g", key, double_value);

      if (i < (size - 1))
        fprintf(stream, ",");
      fprintf(stream,"\n");
    }
    fprintf(stream, "}\n");
    fclose( stream );
  }
  free( filename );
}

void value_export(const value_export_type * value) {
  value_export_txt( value );
  value_export_json( value );
}

void value_export_append( value_export_type * value, const char * key , double double_value) {
  stringlist_append_copy( value->keys, key );
  double_vector_append( value->values, double_value );
}


/*****************************************************************/

UTIL_IS_INSTANCE_FUNCTION( value_export , VALUE_EXPORT_TYPE_ID )
