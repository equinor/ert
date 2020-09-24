/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'rng_manager.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/test_util.h>
#include <ert/util/rng.h>
#include <ert/util/test_work_area.h>
#include <ert/enkf/rng_manager.hpp>

#define MAX_INT 999999

void test_create() {
  rng_manager_type * rng_manager = rng_manager_alloc_load("file/does/not/exist");
  test_assert_NULL( rng_manager );

  rng_manager = rng_manager_alloc_default( );
  test_assert_true( rng_manager_is_instance( rng_manager ));
  rng_manager_free( rng_manager );
}


void test_default( ) {
  rng_manager_type * rng_manager1 = rng_manager_alloc_default( );
  rng_manager_type * rng_manager2 = rng_manager_alloc_default( );
  rng_manager_type * rng_manager3 = rng_manager_alloc_default( );

  rng_type * rng1     = rng_manager_alloc_rng( rng_manager1 );
  rng_type * rng1_0   = rng_manager_iget( rng_manager1, 0 );
  rng_type * rng1_100 = rng_manager_iget( rng_manager1, 100 );

  rng_type * rng2_0   = rng_manager_iget( rng_manager2, 0 );
  rng_type * rng2     = rng_manager_alloc_rng( rng_manager2 );
  rng_type * rng2_100 = rng_manager_iget( rng_manager2, 100 );

  rng_type * rng3_100 = rng_manager_iget( rng_manager3, 100 );
  rng_type * rng3_0   = rng_manager_iget( rng_manager3, 0 );
  rng_type * rng3     = rng_manager_alloc_rng( rng_manager3 );

  test_assert_int_equal( rng_get_int( rng1_0, MAX_INT ), rng_get_int( rng2_0, MAX_INT));
  rng_get_int( rng3_0 , MAX_INT);
  test_assert_int_equal( rng_get_int( rng1_0, MAX_INT ), rng_get_int( rng3_0, MAX_INT));

  test_assert_int_equal( rng_get_int( rng1_100, MAX_INT ), rng_get_int( rng2_100, MAX_INT));
  rng_get_int( rng3_100 , MAX_INT);
  test_assert_int_equal( rng_get_int( rng1_100, MAX_INT ), rng_get_int( rng3_100, MAX_INT));

  test_assert_int_equal( rng_get_int( rng1, MAX_INT ), rng_get_int( rng2, MAX_INT));
  rng_get_int( rng3 , MAX_INT);
  test_assert_int_equal( rng_get_int( rng1, MAX_INT ), rng_get_int( rng3, MAX_INT));

  rng_free( rng1 );
  rng_free( rng2 );
  rng_free( rng3 );
  rng_manager_free( rng_manager3 );
  rng_manager_free( rng_manager2 );
  rng_manager_free( rng_manager1 );
}


void test_state( ) {
  rng_manager_type * rng_manager = rng_manager_alloc_default( );
  ecl::util::TestArea ta("test_rng");
  rng_manager_iget(rng_manager , 100 );
  rng_manager_save_state( rng_manager , "seed.txt");
  test_assert_true( util_file_exists( "seed.txt" ));
  {
    rng_manager_type * rng_manager1 = rng_manager_alloc_load( "seed.txt");
    rng_manager_type * rng_manager2 = rng_manager_alloc_load( "seed.txt");

    rng_type * rng1_0   = rng_manager_iget( rng_manager1, 0 );
    rng_type * rng1_100 = rng_manager_iget( rng_manager1, 100 );

    rng_type * rng2_0   = rng_manager_iget( rng_manager2, 0 );
    rng_type * rng2_100 = rng_manager_iget( rng_manager2, 100 );

    test_assert_int_equal( rng_get_int( rng1_0, MAX_INT ), rng_get_int( rng2_0, MAX_INT));
    test_assert_int_equal( rng_get_int( rng1_100, MAX_INT ), rng_get_int( rng2_100, MAX_INT));

    rng_manager_free( rng_manager2 );
    rng_manager_free( rng_manager1 );
  }
  rng_manager_free( rng_manager );
}

void test_state_restore( ) {
  ecl::util::TestArea ta("restore");
  rng_manager_type * rng_manager1 = rng_manager_alloc_default( );
  rng_manager_save_state( rng_manager1 , "seed.txt");
  rng_type * rng1 = rng_manager_alloc_rng( rng_manager1 );
  {
    rng_manager_type * rng_manager2 = rng_manager_alloc_load("seed.txt");

    rng_type * rng1_0   = rng_manager_iget( rng_manager1, 0 );
    rng_type * rng1_100 = rng_manager_iget( rng_manager1, 100 );

    rng_type * rng2_100 = rng_manager_iget( rng_manager2, 100 );
    rng_type * rng2_0   = rng_manager_iget( rng_manager2, 0 );

    rng_type * rng2 = rng_manager_alloc_rng( rng_manager2 );
    test_assert_int_equal( rng_get_int( rng1_0, MAX_INT ), rng_get_int( rng2_0, MAX_INT));
    test_assert_int_equal( rng_get_int( rng1_100, MAX_INT ), rng_get_int( rng2_100, MAX_INT));
    test_assert_int_equal( rng_get_int( rng1, MAX_INT ), rng_get_int( rng2, MAX_INT));

    rng_free( rng1 );
    rng_free( rng2 );
    rng_manager_free( rng_manager2 );
  }
  rng_manager_free( rng_manager1 );
}



void test_random( ) {
  rng_manager_type * rng_manager1 = rng_manager_alloc_random( );
  rng_manager_type * rng_manager2 = rng_manager_alloc_random( );

  rng_type * rng1_0   = rng_manager_iget( rng_manager1, 0 );
  rng_type * rng1_100 = rng_manager_iget( rng_manager1, 100 );

  rng_type * rng2_0   = rng_manager_iget( rng_manager2, 0 );
  rng_type * rng2_100 = rng_manager_iget( rng_manager2, 100 );

  test_assert_int_not_equal( rng_get_int( rng1_0, MAX_INT ), rng_get_int( rng2_0, MAX_INT));
  test_assert_int_not_equal( rng_get_int( rng1_100, MAX_INT ), rng_get_int( rng2_100, MAX_INT));

  rng_manager_free( rng_manager2 );
  rng_manager_free( rng_manager1 );
}


static void test_alloc() {
  const char * random_seed1 = "apekatterbesting";
  const char * random_seed2 = "apekatterbesxing";

  rng_manager_type * rng_man0 = rng_manager_alloc(random_seed1);
  rng_manager_type * rng_man1 = rng_manager_alloc(random_seed1);
  rng_manager_type * rng_man_odd = rng_manager_alloc(random_seed2);

  rng_type * rng0_0  = rng_manager_iget(rng_man0, 0);
  rng_type * rng0_42 = rng_manager_iget(rng_man0, 42);

  rng_type * rng1_0  = rng_manager_iget(rng_man1, 0);
  rng_type * rng1_42 = rng_manager_iget(rng_man1, 42);

  rng_type * rng_odd_0  = rng_manager_iget(rng_man_odd, 0);
  rng_type * rng_odd_42 = rng_manager_iget(rng_man_odd, 42);

  test_assert_int_equal(rng_get_int(rng0_0, MAX_INT), rng_get_int(rng1_0, MAX_INT));
  test_assert_int_equal(rng_get_int(rng0_42, MAX_INT), rng_get_int(rng1_42, MAX_INT));

  test_assert_int_not_equal(rng_get_int(rng0_0, MAX_INT), rng_get_int(rng_odd_0, MAX_INT));
  test_assert_int_not_equal(rng_get_int(rng0_42, MAX_INT), rng_get_int(rng_odd_42, MAX_INT));

  rng_manager_free(rng_man0);
  rng_manager_free(rng_man1);
  rng_manager_free(rng_man_odd);
}


int main(int argc , char ** argv) {
  test_alloc();
  test_create();
  test_default();
  test_state();
  test_state_restore();
  test_random( );
}

