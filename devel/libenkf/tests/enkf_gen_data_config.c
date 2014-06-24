/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_gen_data_config.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <sys/types.h>
#include <unistd.h>

#include <ert/util/test_util.h>
#include <ert/enkf/gen_data_config.h>


void test_report_steps_param() {

  gen_data_config_type * config = gen_data_config_alloc("KEY" , false);
  test_assert_false( gen_data_config_is_dynamic( config ));
  test_assert_int_equal( 0 , gen_data_config_num_report_step( config ));
  test_assert_false( gen_data_config_has_report_step( config , 0 ));

  /* Add to parameter should fail. */
  gen_data_config_add_report_step( config , 10 );
  test_assert_int_equal( 0 , gen_data_config_num_report_step( config ));
  test_assert_false( gen_data_config_has_report_step( config , 10 ));

  /* Add to parameter should fail. */
  gen_data_config_set_active_report_steps_from_string( config , "0-9,100");
  test_assert_int_equal( 0 , gen_data_config_num_report_step( config ));
  test_assert_false( gen_data_config_has_report_step( config , 10 ));

  
  gen_data_config_free( config );
}


void test_report_steps_dynamic() {
  gen_data_config_type * config = gen_data_config_alloc("KEY" , true);
  test_assert_true( gen_data_config_is_dynamic( config ));
  test_assert_int_equal( 0 , gen_data_config_num_report_step( config ));
  test_assert_false( gen_data_config_has_report_step( config , 0 ));

  gen_data_config_add_report_step( config , 10 );
  test_assert_int_equal( 1 , gen_data_config_num_report_step( config ));
  test_assert_true( gen_data_config_has_report_step( config , 10 ));
  test_assert_int_equal( gen_data_config_iget_report_step( config , 0 ) , 10);

  gen_data_config_add_report_step( config , 10 );
  test_assert_int_equal( 1 , gen_data_config_num_report_step( config ));
  test_assert_true( gen_data_config_has_report_step( config , 10 ));
  

  gen_data_config_add_report_step( config , 5 );
  test_assert_int_equal( 2 , gen_data_config_num_report_step( config ));
  test_assert_true( gen_data_config_has_report_step( config , 10 ));
  test_assert_int_equal( gen_data_config_iget_report_step( config , 0 ) , 5);
  test_assert_int_equal( gen_data_config_iget_report_step( config , 1 ) , 10);
  
  {
    const int_vector_type * active_steps = gen_data_config_get_active_report_steps( config );
    
    test_assert_int_equal( int_vector_iget( active_steps  , 0 ) , 5);
    test_assert_int_equal( int_vector_iget( active_steps  , 1 ) , 10);
  }
  
  gen_data_config_set_active_report_steps_from_string( config , "0-3,7-10,100"); // 0,1,2,3,7,8,9,10,100
  test_assert_int_equal( 9 , gen_data_config_num_report_step( config ));
  test_assert_int_equal( 0 , gen_data_config_iget_report_step( config , 0 ));
  test_assert_int_equal( 3 , gen_data_config_iget_report_step( config , 3));
  test_assert_int_equal( 9 , gen_data_config_iget_report_step( config , 6));
  test_assert_int_equal( 100 , gen_data_config_iget_report_step( config , 8));
  
  gen_data_config_free( config );
}


int main(int argc , char ** argv) {

  test_report_steps_param();
  test_report_steps_dynamic();

  exit(0);
}

