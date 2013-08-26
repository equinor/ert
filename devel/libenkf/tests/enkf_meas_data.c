/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_meas_data.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <unistd.h>

#include <ert/util/test_util.h>
#include <ert/util/bool_vector.h>

#include <ert/enkf/meas_data.h>



void create_test() {
  bool_vector_type * mask = bool_vector_alloc(0 , false);
  bool_vector_iset( mask , 0 , true );
  bool_vector_iset( mask , 10 , true );
  bool_vector_iset( mask , 20 , true );
  bool_vector_iset( mask , 30 , false );
  {
    meas_data_type * meas_data = meas_data_alloc( mask );
    test_assert_int_equal( 3 , meas_data_get_ens_size( meas_data ));

    meas_data_free( meas_data );
  }
  bool_vector_free( mask );
}



int main(int argc , char ** argv) {
  create_test();
  exit(0);
}

