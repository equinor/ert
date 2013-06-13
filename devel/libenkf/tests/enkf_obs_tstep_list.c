/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_obs_tstep_list.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ert/enkf/obs_tstep_list.h>
#include <ert/util/test_util.h>


int main(int argc , char ** argv) {
  obs_tstep_list_type * tstep_list;

  tstep_list = obs_tstep_list_alloc();
  test_assert_true( obs_tstep_list_is_instance( tstep_list ));
  obs_tstep_list_free( tstep_list );
  
  exit(0);
}

