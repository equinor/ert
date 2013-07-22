/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'sched_history_summary.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <stdbool.h>

#include <ert/util/test_util.h>

#include <ert/ecl/ecl_sum.h>

#include <ert/sched/history.h>



int main(int argc, char **argv) {
  char * sum_case = argv[1];
  ecl_sum_type * refcase = ecl_sum_fread_alloc_case( sum_case , ":" );
  history_type * hist = history_alloc_from_refcase( refcase , true );
  
  test_assert_true( history_is_instance( hist ) );

  history_free( hist );
  ecl_sum_free( refcase );
  exit(0);
}
