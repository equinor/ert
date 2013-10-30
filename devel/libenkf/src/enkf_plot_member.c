/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'enkf_plot_member.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <ert/util/util.h>
#include <ert/util/double_vector.h>
#include <ert/util/time_t_vector.h>
#include <ert/util/bool_vector.h>

#include <ert/enkf/enkf_plot_member.h>

#define ENKF_PLOT_MEMBER_ID 6111861

struct enkf_plot_member_struct {
  UTIL_TYPE_ID_DECLARATION;
  double_vector_type        * data;
  double_vector_type        * work;
  time_t_vector_type        * time;
  bool_vector_type          * mask;
};



UTIL_SAFE_CAST_FUNCTION( enkf_plot_member , ENKF_PLOT_MEMBER_ID )
UTIL_IS_INSTANCE_FUNCTION( enkf_plot_member , ENKF_PLOT_MEMBER_ID )
     


static void enkf_plot_member_reset( enkf_plot_member_type * plot_member ) {
  double_vector_reset( plot_member->data );
  time_t_vector_reset( plot_member->time );
  bool_vector_reset( plot_member->mask );
}


enkf_plot_member_type * enkf_plot_member_alloc( ) {
  enkf_plot_member_type * plot_member = util_malloc( sizeof * plot_member);
  UTIL_TYPE_ID_INIT( plot_member , ENKF_PLOT_MEMBER_ID );

  plot_member->data = double_vector_alloc( 0 , 0 );
  plot_member->time = time_t_vector_alloc(-1 , 0);
  plot_member->mask = bool_vector_alloc( false , 0 );
  plot_member->work = double_vector_alloc(0,0);
  enkf_plot_member_reset( plot_member );
  return plot_member;
}


void enkf_plot_member_free( enkf_plot_member_type * plot_member ) {
  double_vector_free( plot_member->data );
  double_vector_free( plot_member->work );
  time_t_vector_free( plot_member->time );
  bool_vector_free( plot_member->mask );
}


bool enkf_plot_member_all_active( const enkf_plot_member_type * plot_member ) {
  bool all_active = true;
  for (int i=0; i < bool_vector_size( plot_member->mask ); i++) 
    all_active = all_active && bool_vector_iget(plot_member->mask , i );

  return all_active;
}


int enkf_plot_member_size( const enkf_plot_member_type * plot_member ) {
  return bool_vector_size( plot_member->mask );
}


void enkf_plot_member_iset( enkf_plot_member_type * plot_member , int index , time_t time , double value) {
  time_t_vector_iset( plot_member->time , index , time );
  double_vector_iset( plot_member->data , index , value );
  bool_vector_iset( plot_member->mask , index , true );
}

double enkf_plot_member_iget_value( const enkf_plot_member_type * plot_member , int index) {
  return double_vector_iget( plot_member->data , index);
}

time_t enkf_plot_member_iget_time( const enkf_plot_member_type * plot_member , int index) {
  return time_t_vector_iget( plot_member->time , index);
}

bool enkf_plot_member_iget_active( const enkf_plot_member_type * plot_member , int index) {
  return bool_vector_iget( plot_member->mask , index );
}


void enkf_plot_member_free__( void * arg ) {
  enkf_plot_member_type * plot_member = enkf_plot_member_safe_cast( arg );
  enkf_plot_member_free( plot_member );
}




void enkf_plot_member_load( enkf_plot_member_type * plot_member , 
                            enkf_node_type * enkf_node , 
                            enkf_fs_type * fs , 
                            const char * user_key , 
                            int iens , 
                            state_enum state , 
                            bool time_mode , 
                            int step1 , int step2) {
  enkf_plot_member_reset( plot_member );
  const time_map_type * time_map = enkf_fs_get_time_map( fs );
  
  if (enkf_node_vector_storage( enkf_node )) {
    enkf_node_user_get_vector(enkf_node , fs , user_key , iens , state , plot_member->work);
    for (int step = 0; step < time_map_get_size(time_map); step++) 
      enkf_plot_member_iset( plot_member , 
                             step , 
                             double_vector_iget( plot_member->work , step ) , 
                             time_map_iget( time_map , step ));
  } else {
    int step;
    node_id_type node_id = {.iens        = iens,
                            .state       = state, 
                            .report_step = 0 };
    
    for (step = step1 ; step <= step2; step++) {
      double value;
      node_id.report_step = step;
      if (enkf_node_user_get(enkf_node , fs , user_key , node_id , &value)) {
        enkf_plot_member_iset( plot_member , 
                               step , 
                               value , 
                               time_map_iget( time_map , step ));
      }
    }

  }
}



