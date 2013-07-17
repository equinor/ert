/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_enkf_obs_tests.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <ert/enkf/enkf_obs.h>
#include <ert/enkf/obs_vector.h>

int main(int argc, char ** argv) {
  enkf_obs_type * enkf_obs = enkf_obs_alloc();

  obs_vector_type * obs_vector = obs_vector_alloc(GEN_OBS, "WHAT", NULL, 0);
  enkf_obs_add_obs_vector(enkf_obs, "PROP", obs_vector);
  
  exit(0);
}

