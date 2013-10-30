/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_plot_member.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ert/util/test_work_area.h>
#include <ert/util/test_util.h>
#include <ert/util/util.h>
#include <ert/util/thread_pool.h>
#include <ert/util/bool_vector.h>
#include <ert/util/arg_pack.h>

#include <ert/enkf/enkf_plot_member.h>



void create_test() {
  enkf_plot_member_type * member = enkf_plot_member_alloc( NULL , 0 );
  test_assert_true( enkf_plot_member_is_instance( member ));
  enkf_plot_member_free( member );
}



void test_iset() {
  enkf_plot_member_type * member = enkf_plot_member_alloc( NULL , 0 );
  enkf_plot_member_iset( member , 10 , 0 , 100 );

  test_assert_int_equal( 11 , enkf_plot_member_size( member  ));
  test_assert_time_t_equal( 0   , enkf_plot_member_iget_time( member , 10 ));
  test_assert_double_equal( 100 , enkf_plot_member_iget_value( member , 10 ));
  {
    for (int i=0; i < (enkf_plot_member_size( member  ) - 1); i++) 
      test_assert_false( enkf_plot_member_iget_active( member , i ));
    
    test_assert_true( enkf_plot_member_iget_active( member , 10 ));
  }
  
  enkf_plot_member_free( member );
}


void test_all_active() {
  enkf_plot_member_type * member = enkf_plot_member_alloc( NULL , 0 );
  test_assert_true( enkf_plot_member_all_active( member ));

  enkf_plot_member_iset( member , 00 , 0 , 100 );
  test_assert_true( enkf_plot_member_all_active( member ));
  
  enkf_plot_member_iset( member , 1 , 0 , 100 );
  test_assert_true( enkf_plot_member_all_active( member ));

  enkf_plot_member_iset( member , 10 , 0 , 100 );
  test_assert_false( enkf_plot_member_all_active( member ));
}



void test_iget() {
  enkf_plot_member_type * member = enkf_plot_member_alloc( NULL , 0 );
  enkf_plot_member_iset( member , 0 , 0 , 0 );
  enkf_plot_member_iset( member , 1 , 1 , 10 );
  enkf_plot_member_iset( member , 2 , 2 , 20 );
  enkf_plot_member_iset( member , 3 , 3 , 30 );
  enkf_plot_member_iset( member , 4 , 4 , 40 );

  enkf_plot_member_iset( member , 6 , 6 , 60 );


  test_assert_int_equal( 7 , enkf_plot_member_size( member ));
  for (int i=0; i < 7; i++) {
    if (i == 5)
      test_assert_false( enkf_plot_member_iget_active( member , i ));
    else {
      test_assert_true( enkf_plot_member_iget_active( member , i ));
      test_assert_time_t_equal( i      , enkf_plot_member_iget_time( member , i ));
      test_assert_double_equal( i * 10 , enkf_plot_member_iget_value( member , i ));
    }
  }
}



int main(int argc , char ** argv) {
  create_test();
  test_iset();
  test_all_active();
  test_iget();
  
  exit(0);
}

