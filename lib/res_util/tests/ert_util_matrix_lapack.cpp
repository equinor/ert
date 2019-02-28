/*
   Copyright (C) 2015  Equinor ASA, Norway.

   The file 'ert_util_matrix_lapack.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <cmath>

#include <ert/util/bool_vector.hpp>
#include <ert/util/test_util.hpp>
#include <ert/util/statistics.hpp>
#include <ert/util/test_work_area.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/rng.hpp>
#include <ert/util/mzran.hpp>
#include <ert/res_util/matrix_lapack.hpp>




void test_det4() {
  matrix_type * m = matrix_alloc(4  , 4 );
  rng_type * rng = rng_alloc(MZRAN , INIT_DEV_URANDOM );
  for (int i=0; i < 10; i++) {
    matrix_random_init( m , rng );
    {
      double det4 = matrix_det4( m );
      double det = matrix_det( m );

      test_assert_double_equal( det , det4 );
    }
  }

  matrix_free( m );
  rng_free( rng );
}


void test_det3() {
  matrix_type * m = matrix_alloc(3  , 3 );
  rng_type * rng = rng_alloc(MZRAN , INIT_DEV_URANDOM );
  matrix_random_init( m , rng );

  {
    double det3 = matrix_det3( m );
    double det = matrix_det( m );

    test_assert_double_equal( det , det3 );
  }

  matrix_free( m );
  rng_free( rng );
}


void test_det2() {
  matrix_type * m = matrix_alloc(2,2);
  rng_type * rng = rng_alloc(MZRAN , INIT_DEV_URANDOM );
  matrix_random_init( m , rng );
  {
    double det2 = matrix_det2( m );
    double det = matrix_det( m );

    test_assert_double_equal( det , det2 );
  }
  matrix_free( m );
  rng_free( rng );
}

void test_dgesvx() {
  matrix_type * m1 = matrix_alloc(3  , 3 );
  matrix_type * m2 = matrix_alloc(3  , 3 );
  matrix_type * b1 = matrix_alloc(3  , 5 );
  matrix_type * b2 = matrix_alloc(3  , 5 );
  matrix_type * b3 = matrix_alloc(3  , 5 );
  rng_type * rng = rng_alloc(MZRAN , INIT_DEV_URANDOM );
  double rcond;
  double epsilon=0.0000000001;
  double diag=1.0;
  {
     matrix_diag_set_scalar(m1, diag);
     matrix_random_init( b1 , rng );
     matrix_assign(b2 , b1);
     matrix_dgesv( m1 , b1  );
     test_assert_true( matrix_similar( b1 , b2 , epsilon ) );

     matrix_diag_set_scalar(m1, diag);
     matrix_random_init( b1 , rng );
     matrix_assign(b2 , b1);
     matrix_dgesvx( m1 , b1, &rcond  );
     test_assert_true( matrix_similar( b1 , b2 , epsilon ) );


     matrix_random_init( m1 , rng );
     matrix_assign(m2 , m1);

     matrix_random_init( b1 , rng );
     matrix_assign(b2 , b1);

     matrix_dgesv( m1 , b1  );
     matrix_dgesvx( m2 , b2, &rcond  );


     matrix_sub(b3,b2,b1);
     matrix_pretty_fprint_submat(b3,"b3","%.*f ",stdout,0,2,0,4) ;

     test_assert_true( matrix_similar( b1 , b2 , epsilon ) );

  }

  matrix_free( m1 );
  matrix_free( m2 );
  matrix_free( b1 );
  matrix_free( b2 );
  rng_free( rng );
}

void test_matrix_similar() {
  rng_type * rng = rng_alloc(MZRAN , INIT_DEV_URANDOM );
  matrix_type * m1 = matrix_alloc( 3 , 3 );
  matrix_type * m2 = matrix_alloc( 3 , 3 );
  double epsilon=0.0000000001;

  matrix_random_init( m1 , rng );
  matrix_assign( m2 , m1 );

  test_assert_true( matrix_similar( m1 , m2, epsilon));

  matrix_iadd(m2,2,2,0.5*epsilon);
  test_assert_true( matrix_similar( m1 , m2, epsilon));

  matrix_iadd(m2,2,2,epsilon);
  test_assert_true( !matrix_similar( m1 , m2, epsilon));

}


int main( int argc , char ** argv) {
  test_det2();
  test_det3();
  test_det4();
  test_dgesvx();
  test_matrix_similar();
  exit(0);
}
