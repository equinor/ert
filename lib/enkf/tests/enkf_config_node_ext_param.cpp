/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_enkf_config_node_gen_data.c' is part of ERT -
   Ensemble based Reservoir Tool.

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

#include <ert/util/test_work_area.h>
#include <ert/util/test_util.h>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/ext_param.hpp>
#include <ert/enkf/ext_param_config.hpp>
#include <ert/enkf/enkf_types.hpp>

enkf_config_node_type * create_config_node__( const char * outfile)
{
  stringlist_type * keys = stringlist_alloc_new( );
  stringlist_append_copy( keys, "KEY1");
  stringlist_append_copy( keys, "KEY2");
  stringlist_append_copy( keys, "KEY3");

  enkf_config_node_type * config_node = enkf_config_node_alloc_EXT_PARAM("key" , keys, outfile);
  stringlist_free( keys );
  return config_node;
}

enkf_config_node_type * create_config_node() {
  return create_config_node__( NULL );
}




void test_create_invalid_keys() {
  stringlist_type * keys = stringlist_alloc_new( );
  stringlist_append_copy( keys, "KEY1");
  stringlist_append_copy( keys, "KEY2");
  stringlist_append_copy( keys, "KEY1");

  test_assert_NULL( enkf_config_node_alloc_EXT_PARAM("key" , keys , NULL) );
  stringlist_free( keys );
}


void test_create_empty_keys() {
  stringlist_type * keys = stringlist_alloc_new( );
  test_assert_NULL( enkf_config_node_alloc_EXT_PARAM("key" , keys , NULL) );
  stringlist_free( keys );
}


void test_create() {
  enkf_config_node_type * config_node = create_config_node( );
  ext_param_config_type * ext_config = (ext_param_config_type *)enkf_config_node_get_ref( config_node );
  test_assert_true( enkf_config_node_is_instance( config_node ));
  test_assert_int_equal( enkf_config_node_get_impl_type( config_node ), EXT_PARAM );
  test_assert_int_equal( enkf_config_node_get_data_size( config_node , 0 ), 3 );
  test_assert_true( ext_param_config_has_key( ext_config , "KEY2"));
  test_assert_false( ext_param_config_has_key( ext_config , "KEY4"));
  enkf_config_node_free( config_node );
}


void test_create_data() {
  enkf_config_node_type * config_node = create_config_node( );
  enkf_node_type * node = enkf_node_alloc( config_node );
  test_assert_true( enkf_node_is_instance( node ));

  ext_param_type * ext_param = (ext_param_type *) enkf_node_value_ptr( node );
  test_assert_false( ext_param_iset( ext_param , -1 , 100 ));
  test_assert_false( ext_param_iset( ext_param ,  3 , 100 ));


  test_assert_true( ext_param_iset( ext_param, 0, 99 ));
  test_assert_double_equal( ext_param_iget( ext_param, 0 ) , 99 );

  test_assert_false( ext_param_key_set( ext_param , "MISSING_KEY" , 100));

  enkf_node_free( node );
  enkf_config_node_free( config_node );
}

void test_forward_write() {
  test_work_area_type * work_area = test_work_area_alloc( "test_json");
  enkf_config_node_type * config_node1 = create_config_node( );
  enkf_config_node_type * config_node2 = create_config_node__( "output/file.json");
  enkf_node_type * node1 = enkf_node_alloc( config_node1 );
  enkf_node_type * node2 = enkf_node_alloc( config_node2 );

  enkf_node_ecl_write( node1, "./" , NULL, 0 );
  enkf_node_ecl_write( node2, "./" , NULL, 0 );
  test_assert_true(util_file_exists( "key.json" ));
  test_assert_true(util_file_exists( "output/file.json" ));

  enkf_node_free( node1 );
  enkf_node_free( node2 );
  enkf_config_node_free( config_node1 );
  enkf_config_node_free( config_node2 );
  test_work_area_free( work_area );
}


void test_write_node( enkf_node_type * node ) {
  enkf_fs_type * fs = enkf_fs_create_fs("mnt" , BLOCK_FS_DRIVER_ID , NULL , true);
  node_id_type node_id = {.report_step = 0 , .iens = 0};
  test_assert_true( enkf_node_store( node , fs, false, node_id) );
  enkf_node_load( node , fs, node_id );
  test_assert_int_equal( enkf_fs_decref( fs ), 0 );
}


void test_read_node( enkf_node_type * node ) {
  enkf_fs_type * fs = enkf_fs_mount("mnt");
  node_id_type node_id = {.report_step = 0 , .iens = 0};
  enkf_node_load( node , fs, node_id );
  enkf_fs_decref( fs );
}


void test_fs() {
  test_work_area_type * work_area = test_work_area_alloc( "test_json");
  enkf_config_node_type * config_node = create_config_node( );
  enkf_node_type * node = enkf_node_alloc( config_node );
  ext_param_type * ext_param = (ext_param_type *) enkf_node_value_ptr( node );
  int data_size = enkf_config_node_get_data_size( config_node, 0);

  for (int i=0; i < data_size; i++)
    ext_param_iset( ext_param , i , i );

  test_write_node( node );
  for (int i=0; i < data_size; i++)
    ext_param_iset( ext_param , i , 99 );

  test_read_node( node );
  for (int i=0; i < data_size; i++)
    test_assert_double_equal( ext_param_iget( ext_param, i) , i );

  enkf_node_free( node );
  enkf_config_node_free( config_node );
  test_work_area_free( work_area );
}


int main( int argc , char **argv ) {
  util_install_signals();
  test_create_invalid_keys();
  test_create_empty_keys();
  test_create();
  test_create_data();
  test_forward_write();
  test_fs();
  exit(0);
}
