/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
   The file 'enkf_plot_blockdata.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <ert/enkf/enkf_plot_blockdata.h>


#define ENKF_PLOT_BLOCKDATA_TYPE_ID 37762063

struct enkf_plot_blockdata_struct {
  UTIL_TYPE_ID_DECLARATION;
  const block_obs_type * block_obs;
};


UTIL_IS_INSTANCE_FUNCTION( enkf_plot_blockdata , ENKF_PLOT_BLOCKDATA_TYPE_ID )


enkf_plot_blockdata_type * enkf_plot_blockdata_alloc( const block_obs_type * block_obs) {
  enkf_plot_blockdata_type * data = util_malloc( sizeof * data );
  UTIL_TYPE_ID_INIT( data , ENKF_PLOT_BLOCKDATA_TYPE_ID );
  data->block_obs = block_obs;
  return data;
}



void enkf_plot_blockdata_free( enkf_plot_blockdata_type * data ) {
  free( data );
}

