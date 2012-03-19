/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'enkf_plot_data.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <double_vector.h>
#include <vector.h>

#include <enkf_fs.h>
#include <enkf_plot_member.h>
#include <enkf_plot_data.h>
#include <member_config.h>



struct enkf_plot_data_struct {
  time_t         start_time;
  vector_type  * ensemble;    // Vector enkf_plot_member_type instance
};
  




void enkf_plot_data_append_member( enkf_plot_data_type * plot_data , member_config_type * member_config ) {
  enkf_plot_member_type * plot_member = enkf_plot_member_alloc( member_config , plot_data->start_time );
  vector_append_owned_ref( plot_data->ensemble , plot_member , enkf_plot_member_free__ );
}


void enkf_plot_data_free( enkf_plot_data_type * plot_data ) {
  vector_free( plot_data->ensemble );
  free( plot_data );
}

enkf_plot_data_type * enkf_plot_data_alloc( time_t start_time) {
  enkf_plot_data_type * plot_data = util_malloc( sizeof * plot_data , __func__ );
  plot_data->start_time = start_time;
  plot_data->ensemble = vector_alloc_new();
  
  return plot_data;
}


void * enkf_plot_data_load__( void *arg ) {
  arg_pack_type * arg_pack            = arg_pack_safe_cast( arg );
  enkf_plot_data_type * plot_data       = arg_pack_iget_ptr( arg_pack , 0 );
  enkf_config_node_type * config_node = arg_pack_iget_ptr( arg_pack , 1 );
  enkf_fs_type * fs                   = arg_pack_iget_ptr( arg_pack , 2 );
  const char * user_key               = arg_pack_iget_ptr( arg_pack , 3 );
  state_enum state                    = arg_pack_iget_int( arg_pack , 4 );
  int step1                           = arg_pack_iget_int( arg_pack , 5 );
  int step2                           = arg_pack_iget_int( arg_pack , 6 );
  int iens1                           = arg_pack_iget_int( arg_pack , 7 );
  int iens2                           = arg_pack_iget_int( arg_pack , 8 );

  enkf_node_type * enkf_node = enkf_node_alloc( config_node ); // Shared node used for all loading.
  
  for (int iens = iens1; iens < iens2; iens++)
    enkf_plot_member_load( vector_iget( plot_data->ensemble , iens) , enkf_node , fs , user_key , state , step1 , step2 );

  enkf_node_free( enkf_node );
  return NULL;
}



void enkf_plot_data_load( enkf_plot_data_type * plot_data , enkf_config_node_type * config_node , enkf_fs_type * fs , const char * user_key , state_enum state , int step1 , int step2) {
  int ens_size    = vector_get_size( plot_data->ensemble );
  int num_threads = 4;
  int block_size  = ens_size / num_threads;
  arg_pack_type ** arg_list = util_malloc( num_threads * sizeof * arg_list , __func__ );
  thread_pool_type * tp = thread_pool_alloc( num_threads , true );
  
  if (block_size == 0)  /* Fewer tasks than threads */
    block_size = 1;
  
  for (int i=0; i < num_threads; i++) {
    int iens1 , iens2;
    arg_list[i] = arg_pack_alloc();
    
    {
      int iens1 = i * block_size;
      int iens2 = iens1 + block_size;
      
      if (iens1 < ens_size) {
        if (iens2 > ens_size)
          iens2 = ens_size;

        arg_pack_append_ptr( arg_list[i] , plot_data );
        arg_pack_append_ptr( arg_list[i] , config_node );
        arg_pack_append_ptr( arg_list[i] , fs );
        arg_pack_append_ptr( arg_list[i] , user_key );
        
        arg_pack_append_int( arg_list[i] , state );
        arg_pack_append_int( arg_list[i] , step1 );
        arg_pack_append_int( arg_list[i] , step2 );
    

        arg_pack_append_int( arg_list[i] , iens1 );
        arg_pack_append_int( arg_list[i] , iens2 );
        
        thread_pool_add_job(tp , enkf_plot_data_load__ , arg_list[i]);
      }
    }
  }
  
  thread_pool_join( tp );
  thread_pool_free( tp );
  for (int i=0; i < num_threads; i++) 
    arg_pack_free( arg_list[i] );
  free( arg_list );
}
