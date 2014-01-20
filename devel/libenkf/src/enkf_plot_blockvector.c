/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
   The file 'enkf_plot_blockvector.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <time.h>
#include <stdbool.h>

#include <ert/util/double_vector.h>
#include <ert/util/vector.h>
#include <ert/util/thread_pool.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/block_obs.h>
#include <ert/enkf/enkf_plot_tvector.h>
#include <ert/enkf/enkf_plot_blockvector.h>


#define ENKF_PLOT_BLOCKVECTOR_TYPE_ID 77362063

struct enkf_plot_blockvector_struct {
  UTIL_TYPE_ID_DECLARATION;
  int size;
  const block_obs_type * block_obs;
};


UTIL_IS_INSTANCE_FUNCTION( enkf_plot_blockvector , ENKF_PLOT_BLOCKVECTOR_TYPE_ID )


enkf_plot_blockvector_type * enkf_plot_blockvector_alloc( const block_obs_type * block_obs ) {
  enkf_plot_blockvector_type * vector = util_malloc( sizeof * vector );
  UTIL_TYPE_ID_INIT( vector , ENKF_PLOT_BLOCKVECTOR_TYPE_ID );
  vector->block_obs = block_obs;
  vector->size = 0;
  return vector;
}


void enkf_plot_blockvector_free( enkf_plot_blockvector_type * vector ) {
  free( vector );
}


int enkf_plot_blockvector_get_size( const enkf_plot_blockvector_type * vector ) { 
  return vector->size;
}
