/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'enkf_plot_member.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <util.h>

#include <member_config.h>
#include <enkf_plot_member.h>
#include <double_vector.h>
#include <time_t_vector.h>


#define ENKF_PLOT_MEMBER_ID 6111861

struct enkf_plot_member_struct {
  UTIL_TYPE_ID_DECLARATION;
  const member_config_type  * member_config;
  double_vector_type        * data;
  time_t_vector_type        * sim_time;
  double_vector_type        * sim_days;
  time_t                      start_time;
};



static UTIL_SAFE_CAST_FUNCTION( enkf_plot_member , ENKF_PLOT_MEMBER_ID )

enkf_plot_member_type * enkf_plot_member_alloc( const member_config_type * member_config , time_t start_time) {
  enkf_plot_member_type * plot_member = util_malloc( sizeof * plot_member , __func__ );
  UTIL_TYPE_ID_INIT( plot_member , ENKF_PLOT_MEMBER_ID )
  plot_member->data     = double_vector_alloc( 0 , 0 );
  plot_member->sim_days = double_vector_alloc( 0 , 0 );
  plot_member->sim_time = time_t_vector_alloc( 0 , 0 );
  plot_member->member_config = member_config;
  return plot_member;
}



void enkf_plot_member_free( enkf_plot_member_type * plot_member ) {
  double_vector_free( plot_member->data );
  double_vector_free( plot_member->sim_days );
  time_t_vector_free( plot_member->sim_time );
  free( plot_member );
}


void enkf_plot_member_free__( void * arg ) {
  enkf_plot_member_type * plot_member = enkf_plot_member_safe_cast( arg );
  enkf_plot_member_free( plot_member );
}


void enkf_plot_member_load( enkf_plot_member_type * plot_member , enkf_node_type * enkf_node , enkf_fs_type * fs , const char * user_key , state_enum state , int step1 , int step2) {
  int iens = member_config_get_iens( plot_member->member_config );
  if (enkf_node_vector_storage( enkf_node )) {
    enkf_node_user_get_vector(enkf_node , fs , user_key , iens , state , plot_member->data);
    time_t_vector_memcpy( plot_member->sim_time , member_config_get_sim_time_ref( plot_member->member_config , fs));
    if (step1 > 0) {
      time_t_vector_idel_block( plot_member->sim_time , 0 , step1 );
      double_vector_idel_block( plot_member->data , 0 , step1 );
    }
  } else {
    int step;
    node_id_type node_id = {.iens        = iens,
                            .state       = state, 
                            .report_step = 0 };

    double_vector_reset( plot_member->data );
    time_t_vector_reset( plot_member->sim_time );

    for (step = step1 ; step <= step2; step++) {
      double value;
      if (enkf_node_user_get(enkf_node , fs , user_key , node_id , &value)) {
        double_vector_append( plot_member->data , value);
        time_t_vector_append( plot_member->sim_time , member_config_iget_sim_time( plot_member->member_config , step , fs ));
      }
    }
  }
}



