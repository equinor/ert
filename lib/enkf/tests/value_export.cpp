/*
   Copyright (C) 2017  Equinor ASA, Norway.

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

#include <iostream>
#include <fstream>

#include <ext/json/cJSON.h>

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>
#include <ert/enkf/value_export.hpp>


void test_create() {
  ecl::util::TestArea ta("value_export");
  value_export_type * export_value = value_export_alloc( "", "parameters");

  test_assert_int_equal( 0 , value_export_size( export_value ));


  test_assert_true( value_export_is_instance( export_value ));

  value_export_txt( export_value );
  test_assert_false( util_file_exists( "parameters.txt" ));

  value_export_json( export_value );
  test_assert_false( util_file_exists( "parameters.json" ));

  value_export_free( export_value );
}



void test_export_json() {
  ecl::util::TestArea ta("value_export_json");
  value_export_type * export_value = value_export_alloc( "path", "parameters");
  util_make_path( "path" );

  value_export_append(export_value, "KEY100", "SUBKEY1", 100);
  value_export_append(export_value, "KEY200", "SUBKEY2", 200);
  value_export_append(export_value, "KEY100", "SUBKEY2", 300);
  value_export_append(export_value, "KEY200", "SUBKEY1", 400);
  value_export_append(export_value, "KEY200", "SUBKEY3", 500);
  test_assert_int_equal( 5 , value_export_size( export_value ));
  value_export_json( export_value );

  test_assert_true( util_file_exists( "path/parameters.json" ));

  std::ifstream f("path/parameters.json");
  std::string strJSON((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());

  cJSON *json = cJSON_Parse(strJSON.c_str());
  test_assert_not_NULL(json);

  const cJSON * key1 = cJSON_GetObjectItemCaseSensitive(json, "KEY100");
  test_assert_true(cJSON_IsObject(key1));

  const cJSON * subkey1 = cJSON_GetObjectItemCaseSensitive(key1, "SUBKEY1");
  test_assert_true(cJSON_IsNumber(subkey1));

  const cJSON * compkey1 = cJSON_GetObjectItemCaseSensitive(json, "KEY100:SUBKEY1");
  test_assert_double_equal(compkey1->valuedouble, 100);

  value_export_free( export_value );
}


void test_export_txt__() {
  ecl::util::TestArea ta("export_txt");
  value_export_type * export_value = value_export_alloc( "", "parameters");
  value_export_append(export_value, "KEY100", "SUBKEY1", 100);
  value_export_append(export_value, "KEY200", "SUBKEY2", 200);
  test_assert_int_equal( 2 , value_export_size( export_value ));

  value_export_txt( export_value );
  value_export_txt__( export_value , "parameters__.txt");
  test_assert_true( util_file_exists( "path/parameters__.txt" ));
  test_assert_true( util_files_equal( "path/parameters__.txt", "path/parameters.txt"));
  value_export_free( export_value );
}


void test_export_txt() {
  ecl::util::TestArea ta("export_txt");
  value_export_type * export_value = value_export_alloc( "path", "parameters");
  util_make_path( "path" );

  value_export_append(export_value, "KEY100", "SUBKEY1", 100);
  value_export_append(export_value, "KEY200", "SUBKEY2", 200);
  test_assert_int_equal( 2 , value_export_size( export_value ));

  value_export_txt( export_value );
  test_assert_true( util_file_exists( "path/parameters.txt" ));


  value_export_txt__( export_value , "path/parameters__.txt");
  test_assert_true( util_file_exists( "path/parameters__.txt" ));
  test_assert_true( util_files_equal( "path/parameters__.txt", "path/parameters.txt"));
  {
    FILE * stream = util_fopen("path/parameters.txt", "r");
    char key1[100],key2[100], subkey1[100], subkey2[100];
    double v1,v2;

    fscanf( stream, "%[^:]:%s %lg %[^:]:%s %lg" , key1, subkey1, &v1, key2, subkey2, &v2);
    fclose( stream );

    test_assert_string_equal( key1, "KEY100");
    test_assert_string_equal( subkey1, "SUBKEY1");
    test_assert_string_equal( key2, "KEY200");
    test_assert_string_equal( subkey2, "SUBKEY2");

    test_assert_double_equal( v1, 100 );
    test_assert_double_equal( v2, 200 );
  }

  value_export_free( export_value );
}



int main(int argc , char ** argv) {
  test_create();
  test_export_txt();
  test_export_json();
  exit(0);
}

