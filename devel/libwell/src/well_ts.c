/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'well_ts.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <stdlib.h>
#include <stdbool.h>
#include <util.h>
#include <vector.h>
#include <well_ts.h>
#include <well_const.h>
#include <well_state.h>



#define WELL_TS_TYPE_ID    6613005
#define WELL_NODE_TYPE_ID  1114652

typedef struct {
  UTIL_TYPE_ID_DECLARATION;
  int                 report_nr;
  time_t              sim_time;  
  well_state_type   * well_state;  // The well_node instance owns the well_state instance.
} well_node_type;


struct well_ts_struct {
  UTIL_TYPE_ID_DECLARATION;
  time_t               min_time;
  time_t               max_time;
  int                  min_report;
  int                  max_report;
  char               * well_name;
  vector_type        * ts;    
};

/******************************************************************/

static well_node_type * well_node_alloc( well_state_type * well_state) {
  well_node_type * node = util_malloc( sizeof * node , __func__ );
  UTIL_TYPE_ID_INIT( node , WELL_NODE_TYPE_ID );
  node->report_nr  = well_state_get_report_nr( well_state );
  node->sim_time   = well_state_get_sim_time( well_state );
  node->well_state = well_state;
  return node;
}


static UTIL_SAFE_CAST_FUNCTION( well_node , WELL_NODE_TYPE_ID )


static void well_node_free( well_node_type * well_node ) {
  well_state_free( well_node->well_state );
  free( well_node );
}

static void well_node_free__( void * arg ) {
  well_node_type * node = well_node_safe_cast( arg );
  well_node_free( node );
}

/*****************************************************************/

static well_ts_type * well_ts_alloc_empty( ) {
  well_ts_type * well_ts = util_malloc( sizeof * well_ts , __func__ );
  UTIL_TYPE_ID_INIT( well_ts , WELL_TS_TYPE_ID );

  well_ts->ts          = vector_alloc_new();
  
  well_ts->min_time   = -1;
  well_ts->max_time   = -1;
  well_ts->min_report = -1;
  well_ts->max_report = -1;
  return well_ts;
}

static UTIL_SAFE_CAST_FUNCTION( well_ts , WELL_TS_TYPE_ID )


well_ts_type * well_ts_alloc( const char * well_name ) {
  well_ts_type * well_ts = well_ts_alloc_empty();
  well_ts->well_name = util_alloc_string_copy( well_name );
  return well_ts;
}






void well_ts_add_well( well_ts_type * well_ts , well_state_type * well_state ) {
  well_node_type * node = well_node_alloc( well_state );
  if (vector_get_size( well_ts->ts ) == 0) 
    vector_append_owned_ref( well_ts->ts , node , well_node_free__ );
  else {
    if (node->sim_time > well_ts->max_time) 
      vector_append_owned_ref( well_ts->ts , node , well_node_free__ );
    else {
      int index = 0;
      well_node_type * inode = vector_iget( well_ts->ts , index );
      while (node->sim_time > inode->sim_time) {
        index++;
        inode = vector_iget( well_ts->ts , index );
      }

      if (node->sim_time == inode->sim_time)
        // Replace
        vector_iset_owned_ref( well_ts->ts , index , node , well_node_free__ );
      else
        vector_insert_owned_ref( well_ts->ts , index , node , well_node_free__ );
    }
  }
  
  if (well_ts->min_time == -1) {
    well_ts->min_time = node->sim_time;
    well_ts->max_time = node->sim_time;

    well_ts->min_report = node->report_nr;
    well_ts->max_report = node->report_nr;
  } else {
    well_ts->min_time = util_time_t_min( well_ts->min_time , node->sim_time );
    well_ts->max_time = util_time_t_max( well_ts->max_time , node->sim_time );

    well_ts->min_report = util_int_min(well_ts->min_report , node->report_nr);
    well_ts->max_report = util_int_max(well_ts->max_report , node->report_nr);
  }
}

static void well_ts_fprintf( const well_ts_type * well_ts , FILE * stream) {
  fprintf(stream,"Report step: %d - %d \n",well_ts->min_report , well_ts->max_report);
}



void well_ts_free( well_ts_type * well_ts ){
  free( well_ts->well_name );
  vector_free( well_ts->ts );
  free( well_ts );
}



void well_ts_free__( void * arg ) {
  well_ts_type * well_ts = well_ts_safe_cast( arg );
  well_ts_free( well_ts );
}


int well_ts_get_size( const well_ts_type * well_ts) {
  return vector_get_size( well_ts->ts );
}

well_state_type * well_ts_iget_state( const well_ts_type * well_ts , int index) {
  return vector_iget( well_ts->ts , index );
}


well_state_type * well_ts_get_state_from_report( const well_ts_type * well_ts , int report_step) {
  well_ts_fprintf( well_ts , stdout );
  if (report_step < well_ts->min_report)
    return NULL;
  else {
    int index = vector_get_size( well_ts->ts ) - 1;
    well_node_type * node;

    /* Linear search ... */
    while (true) {
      node = vector_iget( well_ts->ts , index );
      if (node->report_nr <= report_step)
        break;
      else {
        index--;
        if (index >= 0)
          node = vector_iget( well_ts->ts , index );
        else
          break;
      }
    }
    return node->well_state;
  }
}


well_state_type * well_ts_get_state_from_sim_time( const well_ts_type * well_ts , time_t sim_time) {
  if (sim_time < well_ts->min_time)
    return NULL;
  else {
    int index = vector_get_size( well_ts->ts ) - 1;
    well_node_type * node;

    /* Linear search ... */
    while (true) {
      node = vector_iget( well_ts->ts , index );
      if (node->sim_time <= sim_time)
        break;
      else {
        index--;
        if (index >= 0)
          node = vector_iget( well_ts->ts , index );
        else
          break;
      }
    }
    

    return node->well_state;
    
  }
}

