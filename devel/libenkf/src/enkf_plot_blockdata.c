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
#include <ert/util/arg_pack.h>

#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/block_obs.h>
#include <ert/enkf/obs_vector.h>
#include <ert/enkf/enkf_plot_blockdata.h>
#include <ert/enkf/enkf_plot_blockvector.h>


#define ENKF_PLOT_BLOCKDATA_TYPE_ID 37762063

struct enkf_plot_blockdata_struct {
  UTIL_TYPE_ID_DECLARATION;
  int size;
  const obs_vector_type * obs_vector;
  enkf_plot_blockvector_type ** ensemble;
  arg_pack_type              ** work_arg;
  int                         * sort_perm;
  double_vector_type          * depth;
};


UTIL_IS_INSTANCE_FUNCTION( enkf_plot_blockdata , ENKF_PLOT_BLOCKDATA_TYPE_ID )


enkf_plot_blockdata_type * enkf_plot_blockdata_alloc( const obs_vector_type * obs_vector ) {
  enkf_plot_blockdata_type * data = util_malloc( sizeof * data );
  UTIL_TYPE_ID_INIT( data , ENKF_PLOT_BLOCKDATA_TYPE_ID );
  data->obs_vector = obs_vector;
  data->size = 0;
  data->work_arg = NULL;
  data->ensemble = NULL;
  data->depth = double_vector_alloc(0,0);
  data->sort_perm = NULL;
  
  return data;
}


void enkf_plot_blockdata_free( enkf_plot_blockdata_type * data ) {
  for (int iens=0; iens < data->size; iens++) {
    arg_pack_free( data->work_arg[iens] );
    enkf_plot_blockvector_free( data->ensemble[iens] );
  }
  double_vector_free( data->depth );
  util_safe_free( data->sort_perm );
  free( data->ensemble );
  free( data->work_arg );
  free( data );
}


enkf_plot_blockvector_type * enkf_plot_blockdata_iget( const enkf_plot_blockdata_type * plot_data , int index) {
  return plot_data->ensemble[index];
}



int enkf_plot_blockdata_get_size( const enkf_plot_blockdata_type * data ) {
  return data->size;
}


static void enkf_plot_blockdata_resize( enkf_plot_blockdata_type * plot_blockdata , int new_size ) {
  if (new_size != plot_blockdata->size) {
    int iens;
    
    if (new_size < plot_blockdata->size) {
      for (iens = new_size; iens < plot_blockdata->size; iens++) {
        enkf_plot_blockvector_free( plot_blockdata->ensemble[iens] );
        arg_pack_free( plot_blockdata->work_arg[iens] );
      }
    }

    plot_blockdata->ensemble = util_realloc( plot_blockdata->ensemble , new_size * sizeof * plot_blockdata->ensemble);
    plot_blockdata->work_arg = util_realloc( plot_blockdata->work_arg , new_size * sizeof * plot_blockdata->work_arg);
    
    if (new_size > plot_blockdata->size) {
      for (iens = plot_blockdata->size; iens < new_size; iens++) { 
        plot_blockdata->ensemble[iens] = enkf_plot_blockvector_alloc( plot_blockdata->obs_vector , iens );
        plot_blockdata->work_arg[iens] = arg_pack_alloc();
      }
    } 
    plot_blockdata->size = new_size;
  }
}


static void enkf_plot_blockdata_reset( enkf_plot_blockdata_type * plot_data , int report_step) {
  int iens;
  for (iens = 0; iens < plot_data->size; iens++) 
    arg_pack_clear( plot_data->work_arg[iens] );

  {
    const block_obs_type * block_obs = obs_vector_iget_node( plot_data->obs_vector , report_step );

    util_safe_free( plot_data->sort_perm );
    double_vector_reset( plot_data->depth );
    for (int iobs=0; iobs < block_obs_get_size( block_obs); iobs++) 
      double_vector_append( plot_data->depth , block_obs_iget_depth( block_obs , iobs));

    plot_data->sort_perm = double_vector_alloc_sort_perm( plot_data->depth );
    double_vector_permute( plot_data->depth , plot_data->sort_perm );
  }
}


const double_vector_type * enkf_plot_blockdata_get_depth( const enkf_plot_blockdata_type * plot_data) {
  return plot_data->depth;
}




void enkf_plot_blockdata_load( enkf_plot_blockdata_type * plot_data , 
                               enkf_fs_type * fs , 
                               int report_step , 
                               state_enum state , 
                               const bool_vector_type * input_mask) {
  
  state_map_type * state_map = enkf_fs_get_state_map( fs );
  int ens_size = state_map_get_size( state_map );
  bool_vector_type * mask;
  
  if (input_mask)
    mask = bool_vector_alloc_copy( input_mask );
  else
    mask = bool_vector_alloc( ens_size , false );
  
  state_map_select_matching( state_map , mask , STATE_HAS_DATA );

  enkf_plot_blockdata_resize( plot_data , ens_size );
  enkf_plot_blockdata_reset( plot_data , report_step );
  
  {
    const int num_cpu = 4;
    thread_pool_type * tp = thread_pool_alloc( num_cpu , true );
    for (int iens = 0; iens < ens_size ; iens++) {
      if (bool_vector_iget( mask , iens)) {
        enkf_plot_blockvector_type * vector = enkf_plot_blockdata_iget( plot_data , iens );
        arg_pack_type * work_arg = plot_data->work_arg[iens];

        arg_pack_append_ptr( work_arg , vector );
        arg_pack_append_ptr( work_arg , fs );
        arg_pack_append_int( work_arg , report_step);
        arg_pack_append_int( work_arg , state );
        arg_pack_append_ptr( work_arg , plot_data->sort_perm );

        thread_pool_add_job( tp , enkf_plot_blockvector_load__ , work_arg );
      }
    }
    thread_pool_join( tp );
    thread_pool_free( tp );
  }
  
  bool_vector_free( mask );
}
