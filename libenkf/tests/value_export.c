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
#include <stdbool.h>
#include <stdio.h>

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>

#include <ert/enkf/value_export.h>

void test_create() {
  test_work_area_type * work_area = test_work_area_alloc("value_export");
  value_export_type * export = value_export_alloc( NULL, "parameters");
  test_assert_int_equal( 0 , value_export_size( export ));
  test_assert_true( value_export_is_instance( export ));

  value_export_txt( export );
  test_assert_false( util_file_exists( "parameters.txt" ));

  value_export_json( export );
  test_assert_false( util_file_exists( "parameters.json" ));

  value_export_free( export );
  test_work_area_free( work_area );
}



void test_export_json() {
test_work_area_type * work_area = test_work_area_alloc("value_export");
  value_export_type * export = value_export_alloc( "path", "parameters");
  util_make_path( "path" );

  value_export_append(export, "KEY100", 100);
  value_export_append(export, "KEY200", 200);
  test_assert_int_equal( 2 , value_export_size( export ));
  value_export_json( export );
  test_assert_true( util_file_exists( "path/parameters.json" ));
  value_export_free( export );
  test_work_area_free( work_area );
}


void test_export_txt__() {
  test_work_area_type * work_area = test_work_area_alloc("value_export");
  value_export_type * export = value_export_alloc( NULL, "parameters");
  value_export_append(export, "KEY100", 100);
  value_export_append(export, "KEY200", 200);
  test_assert_int_equal( 2 , value_export_size( export ));

  value_export_txt( export );
  value_export_txt__( export , "parameters__.txt");
  test_assert_true( util_file_exists( "path/parameters__.txt" ));
  test_assert_true( util_files_equal( "path/parameters__.txt", "path/parameters.txt"));
  value_export_free( export );
  test_work_area_free( work_area );
}


void test_export_txt() {
  test_work_area_type * work_area = test_work_area_alloc("value_export");
  value_export_type * export = value_export_alloc( "path", "parameters");
  util_make_path( "path" );

  value_export_append(export, "KEY100", 100);
  value_export_append(export, "KEY200", 200);
  test_assert_int_equal( 2 , value_export_size( export ));

  value_export_txt( export );
  test_assert_true( util_file_exists( "path/parameters.txt" ));

  value_export_txt__( export , "path/parameters__.txt");
  test_assert_true( util_file_exists( "path/parameters__.txt" ));
  test_assert_true( util_files_equal( "path/parameters__.txt", "path/parameters.txt"));
  {
    FILE * stream = util_fopen("path/parameters.txt", "r");
    char key1[100],key2[100];
    double v1,v2;

    fscanf( stream, "%s %lg %s %lg" , key1, &v1, key2, &v2);
    fclose( stream );

    test_assert_string_equal( key1, "KEY100");
    test_assert_string_equal( key2, "KEY200");

    test_assert_double_equal( v1, 100 );
    test_assert_double_equal( v2, 200 );
  }

  value_export_free( export );
  test_work_area_free( work_area );
}



int main(int argc , char ** argv) {
  test_create();
  test_export_txt();
  test_export_json();
  exit(0);
}

