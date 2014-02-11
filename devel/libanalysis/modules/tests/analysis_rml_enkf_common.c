/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
   The file 'analysis_rml_common.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ert/util/test_util.h>
#include <ert/util/rng.h>
#include <ert/util/mzran.h>
#include <ert/util/matrix.h>
#include <ert/util/bool_vector.h>

#include <rml_enkf_common.h>



void test_state() {
  rng_type * rng = rng_alloc( MZRAN , INIT_DEFAULT ); 
  int ens_size    = 10;
  int active_size = 8;
  int rows = 100;
  matrix_type * state = matrix_alloc(1,1);
  bool_vector_type * ens_mask = bool_vector_alloc(ens_size , false);
  matrix_type * A = matrix_alloc( rows , active_size);
  matrix_type * A2 = matrix_alloc( rows, active_size );
  matrix_type * A3 = matrix_alloc( 1,1 );

  for (int i=0; i < active_size; i++)
    bool_vector_iset( ens_mask , i + 1 , true );

  matrix_random_init(A , rng);
  rml_enkf_common_store_state( state , A , ens_mask );

  test_assert_int_equal( matrix_get_rows( state ) , rows );
  test_assert_int_equal( matrix_get_columns( state ) , ens_size );

  {
    int g;
    int a = 0;
    for (g=0; g < ens_size; g++) {
      if (bool_vector_iget( ens_mask , g )) {
        test_assert_true( matrix_columns_equal( state , g , A , a ));
        a++;
      }
    }
  }

  rml_enkf_common_recover_state( state , A2 , ens_mask);
  rml_enkf_common_recover_state( state , A3 , ens_mask);
  test_assert_true( matrix_equal( A , A2 ));
  test_assert_true( matrix_equal( A , A3 ));
  
  bool_vector_free( ens_mask );
  matrix_free( state );
  matrix_free( A );
}


int main(int argc , char ** argv) {
  

  exit(0);
}

