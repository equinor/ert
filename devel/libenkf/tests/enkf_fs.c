/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_fs.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <stdio.h>
#include <stdbool.h>

#include <ert/util/test_util.h>
#include <ert/enkf/ert_test_context.h>
#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/gen_kw.h>



void test_mount() {
  test_assert_false( enkf_fs_exists( "mnt" ));
  enkf_fs_create_fs("mnt" , BLOCK_FS_DRIVER_ID , NULL );
  bool read_only           = false;
  test_assert_true( enkf_fs_exists( "mnt" ));
  {
    enkf_fs_type * fs = enkf_fs_mount( "mnt" , read_only );
    test_assert_true( util_file_exists("mnt/mnt.lock"));
    test_assert_true( enkf_fs_is_instance( fs ));
    enkf_fs_decref( fs );
    test_assert_false( util_file_exists("mnt/mnt.lock"));
  }
}


void test_refcount() {
  enkf_fs_create_fs("mnt" , BLOCK_FS_DRIVER_ID , NULL );
  {
    enkf_fs_type * fs = enkf_fs_mount( "mnt" , false );
    test_assert_int_equal( 1 , enkf_fs_get_refcount( fs ));
    enkf_fs_decref( fs );
  }
}

void test_read_only() {
  enkf_fs_create_fs("mnt" , BLOCK_FS_DRIVER_ID , NULL );
  {
    {
      enkf_fs_type * fs_false = enkf_fs_mount( "mnt" , false );
      test_assert_false(enkf_fs_is_read_only(fs_false));
      test_assert_true( util_file_exists("mnt/mnt.lock"));
      enkf_fs_decref( fs_false );
    }

    {
      enkf_fs_type * fs_true = enkf_fs_mount( "mnt" , true );
      test_assert_true(enkf_fs_is_read_only(fs_true));
      test_assert_false( util_file_exists("mnt/mnt.lock"));
      enkf_fs_decref( fs_true );
    }
  }
}


void create_and_initalize_case(enkf_main_type * enkf_main, const char * new_case) {

  enkf_fs_type * initialized_case = enkf_main_mount_alt_fs(enkf_main, new_case, false, true);
  enkf_main_set_fs(enkf_main, initialized_case, NULL);
  test_assert_string_equal(enkf_main_get_current_fs(enkf_main), enkf_fs_get_case_name(initialized_case));

  int ens_size = enkf_main_get_ensemble_size(enkf_main);
  stringlist_type * param_list = stringlist_alloc_new();
  stringlist_append_copy(param_list, "MULTFLT");
  enkf_main_initialize_from_scratch(enkf_main, param_list, 0, ens_size-1, INIT_FORCE);

  enkf_fs_decref(initialized_case);
}


void test_invalidate_cache(ert_test_context_type * test_context) {
  enkf_main_type * enkf_main = ert_test_context_get_main(test_context);
  test_assert_not_NULL(enkf_main);

  const char * case1 = "test_case1";
  const char * case2 = "test_case2";
  create_and_initalize_case(enkf_main, case1);
  create_and_initalize_case(enkf_main, case2);

  int ensemble_size = enkf_main_get_ensemble_size(enkf_main);

  for (int iens = 0; iens < ensemble_size; ++iens) {

    double val1;
    double val2;

    node_id_type node_id = {.report_step = 0 ,
                            .iens = iens,
                            .state = ANALYZED };

    {
      enkf_main_select_fs(enkf_main, case1);
      test_assert_string_equal(enkf_main_get_current_fs(enkf_main),  case1);

      enkf_state_type * enkf_state = enkf_main_iget_state(enkf_main, iens);
      enkf_node_type * gen_kw_node = enkf_state_get_node(enkf_state, "MULTFLT");
      test_assert_not_NULL(gen_kw_node);

      test_assert_true(enkf_node_try_load(gen_kw_node, enkf_main_get_fs(enkf_main), node_id));
      gen_kw_type * kw_case1 = enkf_node_value_ptr(gen_kw_node);
      val1 = gen_kw_data_iget( kw_case1 , 0 , false );
    }

    {
      enkf_main_select_fs(enkf_main, case2);
      test_assert_string_equal(enkf_main_get_current_fs(enkf_main),  case2);

      enkf_state_type * enkf_state = enkf_main_iget_state(enkf_main, iens);
      enkf_node_type * gen_kw_node = enkf_state_get_node(enkf_state, "MULTFLT");
      test_assert_not_NULL(gen_kw_node);

      test_assert_true(enkf_node_try_load(gen_kw_node, enkf_main_get_fs(enkf_main), node_id));
      gen_kw_type * kw_case2 = enkf_node_value_ptr(gen_kw_node);
      val2 = gen_kw_data_iget( kw_case2, 0 , false );
    }

    test_assert_double_not_equal(val1, val2);
    printf("Realization nr %d: MULTFLT value from test_case 1 is %f, MULTFLT value from test_case2 is %f\n", iens+1, val1, val2);
  }
}



int main(int argc, char ** argv) {
  const char * config_file = argv[1];
  test_assert_not_NULL(config_file);
  ert_test_context_type * test_context = ert_test_context_alloc("test_context_enkf_fs_test" , config_file , NULL);

  test_mount();
  test_refcount();
  test_read_only();
  test_invalidate_cache(test_context);

  ert_test_context_free(test_context);
  exit(0);
}
