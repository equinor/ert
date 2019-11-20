/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'regression.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <math.h>

#include <ert/util/util.hpp>

#include <ert/res_util/matrix.hpp>
#include <ert/res_util/matrix_blas.hpp>
#include <ert/res_util/matrix_lapack.hpp>
#include <ert/res_util/regression.hpp>


/**
   Performs an ordinary least squares estimation of the parameter
   vector beta.

   beta = inv(X' . X) . X' . y
*/

void regression_augmented_OLS( const matrix_type * X , const matrix_type * Y , const matrix_type* Z, matrix_type * beta) {
  /*
    Solves the following especial augmented regression problem:

    [Y ; 0] = [X ; Z] beta + epsilon

    where 0 is the zero matrix of same size as Y.

    The solution to this OLS is:

     inv(X'X + Z'Z) * X' * Y

    The semicolon denotes row concatenation and the apostrophe the transpose.

   */
  int nvar = matrix_get_columns( X );
  matrix_type * Xt   = matrix_alloc_transpose( X );
  matrix_type * Xinv = matrix_alloc( nvar ,  nvar);
  matrix_matmul( Xinv , Xt , X );

  matrix_type * Zt  = matrix_alloc_transpose( Z );
  matrix_type * ZtZ = matrix_alloc( nvar ,  nvar);
  matrix_matmul( ZtZ , Zt , Z );

  // Xinv <- X'X + Z'Z
  matrix_inplace_add(Xinv, ZtZ);

  // Sometimes the inversion fails - add a small regularization to diagonal
  for (int i = 0; i < nvar; ++i)
    matrix_iadd(Xinv, i, i, 1e-10);

  matrix_inv( Xinv ); // Xinv is always invertible
  {
    matrix_type * tmp = matrix_alloc_matmul( Xinv , Xt );
    matrix_matmul( beta , tmp , Y );
    matrix_free( tmp );
  }

  matrix_free( Xt );
  matrix_free( Xinv );
  matrix_free( Zt );
  matrix_free( ZtZ );
}
