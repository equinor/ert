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

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <ext/json/cJSON.h>

#include <ert/util/util.h>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>
#include <ert/enkf/value_export.hpp>


namespace {

// This is super ugly but I could not find any way to do this without using a
// a global variable. Of course the correct thing to do would be to implement
// a new walk_directory that either collects first and return (include
// max_depth, remove callbacks), or behaves like an iterator. But that would
// require quite some efforts, and it's probably not worth it for just one
// test. Especially considering that C++17 includes the filesystem library that
// would make this trivial (but it's a bit unclear what compiler we should take
// as a reference)

std::vector<std::string> storage;

void file_cb(
    char const* path,
    char const* name,
    void *
) {
    storage.push_back(std::string(path) + UTIL_PATH_SEP_STRING + name);
}

bool dir_cb (
    char const* path,
    char const* name,
    int,
    void *
) {
    storage.push_back(std::string(path) + UTIL_PATH_SEP_STRING + name);
    return false;
}

std::vector<std::string> directory_tree(std::string const& root) {
    storage.clear();
    util_walk_directory(
        root.c_str(),
        file_cb,
        nullptr,
        dir_cb,
        nullptr);
    return storage; // this creates a copy of storage
}

} /* unnamed namespace */


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
  auto const strJSON = std::string(
      std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());

  cJSON *json = cJSON_Parse(strJSON.c_str());
  test_assert_not_NULL(json);

  const cJSON * key1 = cJSON_GetObjectItemCaseSensitive(json, "KEY100");
  test_assert_true(cJSON_IsObject(key1));

  const cJSON * subkey1 = cJSON_GetObjectItemCaseSensitive(key1, "SUBKEY1");
  test_assert_true(cJSON_IsNumber(subkey1));

  const cJSON * compkey1 = cJSON_GetObjectItemCaseSensitive(json, "KEY100:SUBKEY1");
  test_assert_double_equal(compkey1->valuedouble, 100);

  // Export again with more values
  value_export_append(export_value, "KEY300", "SUBKEY1", 600);
  test_assert_int_equal( 6 , value_export_size( export_value ));
  value_export_json( export_value );
  std::ifstream f2("path/parameters.json");
  auto const strJSON2 = std::string(
      std::istreambuf_iterator<char>(f2), std::istreambuf_iterator<char>());
  test_assert_true(strJSON2.size() > strJSON.size());

  auto tree = directory_tree("path");
  test_assert_size_t_equal(tree.size(), 2);
  std::sort(
      std::begin(tree),
      std::end(tree),
      [](std::string const& l, std::string const& r) -> bool {
          return l.size() < r.size();
      });
  test_assert_string_equal(tree[0].c_str(), "path/parameters.json");
  test_assert_string_equal(
      tree[1].substr(0, 30).c_str(),
      "path/parameters.json_backup_20"); // Fix this in 80 years

  value_export_free( export_value );
}


void test_export_txt__() {
  ecl::util::TestArea ta("export_txt__");
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


  // Export again with more values
  value_export_append(export_value, "KEY300", "SUBKEY1", 600);
  test_assert_int_equal( 3 , value_export_size( export_value ));
  value_export_txt( export_value );

  auto tree = directory_tree("path");
  test_assert_size_t_equal(tree.size(), 3); // there is also parameters__.txt
  std::sort(
      std::begin(tree),
      std::end(tree),
      [](std::string const& l, std::string const& r) -> bool {
          return l.size() < r.size();
      });

  test_assert_string_equal(tree[0].c_str(), "path/parameters.txt");
  test_assert_string_equal(
      tree[2].substr(0, 29).c_str(),
      "path/parameters.txt_backup_20"); // Fix this in 80 years

  test_assert_false(util_files_equal(tree[0].c_str(), tree[1].c_str()));
  test_assert_true(util_files_equal(tree[2].c_str(), tree[1].c_str()));
  value_export_free( export_value );
}



int main(int argc , char ** argv) {
  test_create();
  test_export_txt();
  test_export_json();
  exit(0);
}

