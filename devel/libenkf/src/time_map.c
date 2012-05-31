/*
   Copyright (C) 2011  Statoil ASA, Norway. 
   The file 'time_map.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#define  _GNU_SOURCE   /* Must define this to get access to pthread_rwlock_t */
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>

#include <util.h>
#include <time_t_vector.h>

#include <ecl_sum.h>

#include <time_map.h>

#define DEFAULT_TIME  -1


struct time_map_struct {
  time_t_vector_type * map;
  pthread_rwlock_t     rw_lock;
};


time_map_type * time_map_alloc( ) {
  time_map_type * map = util_malloc( sizeof * map , __func__);
  map->map = time_t_vector_alloc(0 , DEFAULT_TIME );
  pthread_rwlock_init( &map->rw_lock , NULL);
  return map;
}


void time_map_free( time_map_type * map ) {
  time_t_vector_free( map->map );
  free( map );
}


/**
   Must hold the write lock. 
*/

static void time_map_update__( time_map_type * map , int step , time_t time) {
  time_t current_time = time_t_vector_safe_iget( map->map , step);
  if (current_time == DEFAULT_TIME)
    time_t_vector_iset( map->map , step , time );
  else {
    if (current_time != time)
      util_abort("%s: time mismatch for step:%d \n",__func__ , step );
  }
}


static void time_map_summary_update__( time_map_type * map , const ecl_sum_type * ecl_sum) {
  int first_step = ecl_sum_get_first_report_step( ecl_sum );
  int last_step  = ecl_sum_get_last_report_step( ecl_sum );
  int step;

  for (step = first_step; step <= last_step; step++) {
    if (ecl_sum_has_report_step(ecl_sum , step)) {
      time_t time = ecl_sum_get_report_time( ecl_sum , step ); 
      time_map_update__( map , step , time );
    }
  }
}


static time_t time_map_iget__( const time_map_type * map , int step ) {
  return time_t_vector_safe_iget( map->map , step );
}



void time_map_update( time_map_type * map , int step , time_t time) {
  pthread_rwlock_wrlock( &map->rw_lock );
  time_map_update__( map , step , time );
  pthread_rwlock_unlock( &map->rw_lock );
}


void time_map_summary_update( time_map_type * map , const ecl_sum_type * ecl_sum) {
  pthread_rwlock_wrlock( &map->rw_lock );
  time_map_summary_update__( map , ecl_sum );
  pthread_rwlock_unlock( &map->rw_lock );
}

