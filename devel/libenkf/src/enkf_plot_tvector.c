/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'enkf_plot_tvector.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ert/enkf/enkf_plot_tvector.h>

#define ENKF_PLOT_TVECTOR_ID 6111861

struct enkf_plot_tvector_struct {
  UTIL_TYPE_ID_DECLARATION;
  double_vector_type        * data;
  double_vector_type        * work;
  time_t_vector_type        * time;
  bool_vector_type          * mask;
};



UTIL_SAFE_CAST_FUNCTION( enkf_plot_tvector , ENKF_PLOT_TVECTOR_ID )
UTIL_IS_INSTANCE_FUNCTION( enkf_plot_tvector , ENKF_PLOT_TVECTOR_ID )
     


static void enkf_plot_tvector_reset( enkf_plot_tvector_type * plot_tvector ) {
  double_vector_reset( plot_tvector->data );
  time_t_vector_reset( plot_tvector->time );
  bool_vector_reset( plot_tvector->mask );
}


enkf_plot_tvector_type * enkf_plot_tvector_alloc( ) {
  enkf_plot_tvector_type * plot_tvector = util_malloc( sizeof * plot_tvector);
  UTIL_TYPE_ID_INIT( plot_tvector , ENKF_PLOT_TVECTOR_ID );

  plot_tvector->data = double_vector_alloc( 0 , 0 );
  plot_tvector->time = time_t_vector_alloc(-1 , 0);
  plot_tvector->mask = bool_vector_alloc( false , 0 );
  plot_tvector->work = double_vector_alloc(0,0);
  enkf_plot_tvector_reset( plot_tvector );
  return plot_tvector;
}


void enkf_plot_tvector_free( enkf_plot_tvector_type * plot_tvector ) {
  double_vector_free( plot_tvector->data );
  double_vector_free( plot_tvector->work );
  time_t_vector_free( plot_tvector->time );
  bool_vector_free( plot_tvector->mask );
}


bool enkf_plot_tvector_all_active( const enkf_plot_tvector_type * plot_tvector ) {
  bool all_active = true;
  for (int i=0; i < bool_vector_size( plot_tvector->mask ); i++) 
    all_active = all_active && bool_vector_iget(plot_tvector->mask , i );

  return all_active;
}


int enkf_plot_tvector_size( const enkf_plot_tvector_type * plot_tvector ) {
  return bool_vector_size( plot_tvector->mask );
}


void enkf_plot_tvector_iset( enkf_plot_tvector_type * plot_tvector , int index , time_t time , double value) {
  time_t_vector_iset( plot_tvector->time , index , time );
  double_vector_iset( plot_tvector->data , index , value );
  bool_vector_iset( plot_tvector->mask , index , true );
}

double enkf_plot_tvector_iget_value( const enkf_plot_tvector_type * plot_tvector , int index) {
  return double_vector_iget( plot_tvector->data , index);
}

time_t enkf_plot_tvector_iget_time( const enkf_plot_tvector_type * plot_tvector , int index) {
  return time_t_vector_iget( plot_tvector->time , index);
}

bool enkf_plot_tvector_iget_active( const enkf_plot_tvector_type * plot_tvector , int index) {
  return bool_vector_iget( plot_tvector->mask , index );
}


void enkf_plot_tvector_free__( void * arg ) {
  enkf_plot_tvector_type * plot_tvector = enkf_plot_tvector_safe_cast( arg );
  enkf_plot_tvector_free( plot_tvector );
}




void enkf_plot_tvector_load( enkf_plot_tvector_type * plot_tvector , 
                            enkf_node_type * enkf_node , 
                            enkf_fs_type * fs , 
                            const char * user_key , 
                            int iens , 
                            state_enum state , 
                            bool time_mode , 
                            int step1 , int step2) {
  enkf_plot_tvector_reset( plot_tvector );
  time_map_type * time_map = enkf_fs_get_time_map( fs );
  
  if (enkf_node_vector_storage( enkf_node )) {
    enkf_node_user_get_vector(enkf_node , fs , user_key , iens , state , plot_tvector->work);
    for (int step = 0; step < time_map_get_size(time_map); step++) 
      enkf_plot_tvector_iset( plot_tvector , 
                             step , 
                             double_vector_iget( plot_tvector->work , step ) , 
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
        enkf_plot_tvector_iset( plot_tvector , 
                               step , 
                               value , 
                               time_map_iget( time_map , step ));
      }
    }

  }
}



