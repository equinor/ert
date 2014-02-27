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
#include <ert/enkf/obs_vector.h>
#include <ert/enkf/enkf_node.h>
#include <ert/enkf/enkf_plot_tvector.h>
#include <ert/enkf/enkf_plot_blockvector.h>


#define ENKF_PLOT_BLOCKVECTOR_TYPE_ID 77362063

struct enkf_plot_blockvector_struct {
  UTIL_TYPE_ID_DECLARATION;
  int iens;
  double_vector_type * data;
  const obs_vector_type * obs_vector;
};


UTIL_IS_INSTANCE_FUNCTION( enkf_plot_blockvector , ENKF_PLOT_BLOCKVECTOR_TYPE_ID )


enkf_plot_blockvector_type * enkf_plot_blockvector_alloc( const obs_vector_type * obs_vector , int iens) {
  enkf_plot_blockvector_type * vector = util_malloc( sizeof * vector );
  UTIL_TYPE_ID_INIT( vector , ENKF_PLOT_BLOCKVECTOR_TYPE_ID );
  vector->obs_vector = obs_vector;
  vector->data = double_vector_alloc(0,0);
  vector->iens = iens;
  return vector;
}


void enkf_plot_blockvector_free( enkf_plot_blockvector_type * vector ) {
  double_vector_free( vector->data );
  free( vector );
}


int enkf_plot_blockvector_get_size( const enkf_plot_blockvector_type * vector ) { 
  return double_vector_size( vector->data );
}

double enkf_plot_blockvector_iget( const enkf_plot_blockvector_type * vector , int index)  {
  return double_vector_iget( vector->data , index );
}


void enkf_plot_blockvector_reset( enkf_plot_blockvector_type * vector ) { 
  double_vector_reset( vector->data );
}


void enkf_plot_blockvector_load( enkf_plot_blockvector_type * vector , enkf_fs_type * fs , int report_step , state_enum state, const int * sort_perm) {
  enkf_plot_blockvector_reset( vector );
  {
    const enkf_config_node_type * config_node = obs_vector_get_config_node( vector->obs_vector );
    node_id_type node_id = { .report_step = report_step , 
                             .state       = state , 
                             .iens        = vector->iens };

    enkf_node_type * data_node;
    if (enkf_config_node_get_impl_type( config_node ) == CONTAINER)
      data_node = enkf_node_alloc_private_container( config_node );
    else
      data_node = enkf_node_alloc( config_node );
    
    if (enkf_node_try_load( data_node , fs , node_id )) {
      const block_obs_type * block_obs = obs_vector_iget_node( vector->obs_vector , report_step );
      for (int i=0; i < block_obs_get_size( block_obs ); i++) 
        double_vector_append(vector->data , block_obs_iget_data( block_obs , enkf_node_value_ptr( data_node ) , i , node_id));
      
      double_vector_permute( vector->data , sort_perm );
    }
    enkf_node_free( data_node );
  }
}



void * enkf_plot_blockvector_load__( void * arg ) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  enkf_plot_blockvector_type * vector  = arg_pack_iget_ptr( arg_pack , 0);
  enkf_fs_type * fs = arg_pack_iget_ptr( arg_pack , 1 );
  int report_step = arg_pack_iget_int( arg_pack , 2 );
  state_enum state = arg_pack_iget_int( arg_pack , 3 );
  const int * sort_perm = arg_pack_iget_ptr( arg_pack , 4);

  enkf_plot_blockvector_load( vector , fs , report_step , state , sort_perm);
  return NULL;
}
