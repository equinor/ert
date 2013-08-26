/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_state_map.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <ert/util/arg_pack.h>

#include <ert/enkf/state_map.h>


void create_test() {
  state_map_type * state_map = state_map_alloc();
  test_assert_true( state_map_is_instance( state_map ));
  state_map_free( state_map );
}



int main(int argc , char ** argv) {
  create_test();
  
  exit(0);
}

