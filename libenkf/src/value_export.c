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

#include <ert/util/stringlist.h>
#include <ert/util/double_vector.h>

#include <ert/enkf/value_export.h>

#define VALUE_EXPORT_TYPE_ID     5741761


struct value_export_struct {
  UTIL_TYPE_ID_DECLARATION;
  char * directory;
  char * base_name;

  stringlist_type * keys;
  double_vector_type * values;
};



value_export_type * value_export_alloc(const char * directory, const char * base_name) {
  value_export_type * export = util_malloc( sizeof * export );
  UTIL_TYPE_ID_INIT( export , VALUE_EXPORT_TYPE_ID );
  export->directory = util_alloc_string_copy( directory );
  export->base_name = util_alloc_string_copy( base_name );

  export->keys = stringlist_alloc_new( );
  export->values = double_vector_alloc(0,0);
  return export;
}


void value_export_free(value_export_type * export) {
  stringlist_free( export->keys );
  double_vector_free( export->values );
  free( export->directory );
  free( export->base_name );
  free( export );
}

int value_export_size( const value_export_type * export) {
  return double_vector_size( export->values );
}


void value_export_txt__(const value_export_type * export, const char * filename) {
  const int size = double_vector_size( export->values );
  if (size > 0) {
    FILE * stream = util_fopen( filename , "w");

    for (int i=0; i < size; i++) {
      const char * key          = stringlist_iget( export->keys, i );
      double value              = double_vector_iget( export->values, i );
      fprintf(stream, "%s %g\n", key, value);
    }
    fclose( stream );
  }
}

void value_export_txt(const value_export_type * export) {
  const int size = double_vector_size( export->values );
  if (size > 0) {
    char * filename = util_alloc_filename( export->directory , export->base_name, "txt");
    value_export_txt__( export, filename );
    free( filename );
  }
}

void value_export_json(const value_export_type * export) {
  const int size = double_vector_size( export->values );
  if (size > 0) {
    char * filename = util_alloc_filename( export->directory , export->base_name, "json");
    FILE * stream = util_fopen( filename , "w");
    fprintf(stream, "{\n");
    for (int i=0; i < size; i++) {
      const char * key          = stringlist_iget( export->keys, i );
      double value              = double_vector_iget( export->values, i );
      fprintf(stream, "\"%s\" : %g", key, value);
      if (i < (size - 1))
        fprintf(stream, ",");
      fprintf(stream,"\n");
    }
    fprintf(stream, "}\n");
    fclose( stream );
    free( filename );
  }
}

void value_export(const value_export_type * export) {
  value_export_txt( export );
  value_export_json( export );
}

void value_export_append( value_export_type * export, const char * key , double value) {
  stringlist_append_copy( export->keys, key );
  double_vector_append( export->values, value );
}


/*****************************************************************/

UTIL_IS_INSTANCE_FUNCTION( value_export , VALUE_EXPORT_TYPE_ID )
