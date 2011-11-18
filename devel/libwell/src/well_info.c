/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'well_info.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <util.h>
#include <hash.h>
#include <int_vector.h>
#include <ecl_intehead.h>
#include <ecl_file.h>
#include <ecl_kw.h>
#include <ecl_kw_magic.h>
#include <ecl_util.h>

#include <well_const.h>
#include <well_conn.h>
#include <well_state.h>
#include <well_info.h>
#include <well_ts.h>
#include <stringlist.h>

/*
  The library libwell contains functionality to read and interpret
  some of the well related keywords from an ECLIPSE restart
  file. Roughly speaking the implementation is spread between three
  datatypes:

    well_info_type: This is the container type which holds information
       about all the wells; at all times.

    well_ts_type: The status and properties of a well can typically
       change throughout the simulation; the datatype well_ts_type
       contains a time series for one well.

    well_state_type: The well_state_type datatype contains the
       state/properties of one well at one particular instant of time.

  Limitations

     Read-only: The well properties for ECLIPSE is specified through
       the SCHEDULE section of the ECLIPSE datafile; i.e. the
       information found in restart files is only for
       reporting/visaulization+++, and can not be used to alter the
       simulation.

     segmented wells: The segment information for multi segment wells
       is completely ignored.  

     lgr: Well information from lgr's is ignored; the code 'handles' a
       restart file with lgr - but only the properties from the global
       grid are considered.
*/

/*
  Usage:
  ------


*/


#define WELL_INFO_TYPE_ID 91777451



struct well_info_struct {
  hash_type       * wells;                /* Hash table of well_ts_type instances; indexed by well name. */
  stringlist_type * well_names;           /* A list of all the well names. */
};



well_info_type * well_info_alloc() {
  well_info_type * well_info = util_malloc( sizeof * well_info , __func__ );
  well_info->wells      = hash_alloc();
  well_info->well_names = stringlist_alloc_new();
  return well_info;
}


bool well_info_has_well( well_info_type * well_info , const char * well_name ) {
  return hash_has_key( well_info->wells , well_name );
}

well_ts_type * well_info_get_ts( const well_info_type * well_info , const char *well_name) {
  return hash_get( well_info->wells , well_name );
}

static void well_info_add_new_ts( well_info_type * well_info , const char * well_name) {
  well_ts_type * well_ts = well_ts_alloc( well_name ) ;
  hash_insert_hash_owned_ref( well_info->wells , well_name , well_ts , well_ts_free__);
  stringlist_append_ref( well_info->well_names , well_name );
}

static void well_info_add_state( well_info_type * well_info , well_state_type * well_state) {
  const char * well_name = well_state_get_name( well_state );
  if (!well_info_has_well( well_info , well_name)) 
    well_info_add_new_ts( well_info , well_name );
  
  {
    well_ts_type * well_ts = well_info_get_ts( well_info , well_name );
    well_ts_add_well( well_ts , well_state );
  }
}  

/**
   This function assumes that (sub)select_block() has been used on the
   ecl_file instance @rst_file; and the function will load well
   information from the first block available in the file only. To
   load all the well information from a unified restart file it is
   easier to use the well_info_add_UNRST_wells() function; which works
   by calling this function repeatedly.  
*/

void well_info_add_wells( well_info_type * well_info , ecl_file_type * rst_file , int report_nr , int grid_nr) {
  ecl_intehead_type * header = ecl_intehead_alloc( ecl_file_iget_named_kw( rst_file , INTEHEAD_KW , grid_nr ));
  {
    int well_nr;
    
    for (well_nr = 0; well_nr < header->num_wells; well_nr++) {
      well_state_type * well_state = well_state_alloc( rst_file , header , report_nr , grid_nr , well_nr );
      well_info_add_state( well_info , well_state );
    }
  }
  ecl_intehead_free( header );
}


void well_info_add_UNRST_wells( well_info_type * well_info , ecl_file_type * rst_file, int grid_nr) {
  {
    int num_blocks = ecl_file_get_num_named_kw( rst_file , SEQNUM_KW );
    for (int block_nr = 0; block_nr < num_blocks; block_nr++) {
      ecl_file_push_block( rst_file );      // <-------------------------------------------------------
      {                                                                                               //
        ecl_file_subselect_block( rst_file , SEQNUM_KW , block_nr );                                  //  Ensure that the status 
        {                                                                                             //  is not changed as a side 
          const ecl_kw_type * seqnum_kw = ecl_file_iget_named_kw( rst_file , SEQNUM_KW , 0);          //  effect.
          int report_nr = ecl_kw_iget_int( seqnum_kw , 0 );                                           // 
          well_info_add_wells( well_info , rst_file , report_nr , grid_nr);                           // 
        }                                                                                             //
      }                                                                                               //
      ecl_file_pop_block( rst_file );       // <-------------------------------------------------------           
    }
  }
}

void well_info_load_file( well_info_type * well_info , const char * filename) {
  int report_nr;
  ecl_file_enum file_type = ecl_util_get_file_type( filename , NULL , &report_nr);
  if ((file_type == ECL_RESTART_FILE) || (file_type == ECL_UNIFIED_RESTART_FILE))
  {
    ecl_file_type * ecl_file = ecl_file_open( filename );
    int grid_nr = 0;

    if (file_type == ECL_RESTART_FILE)
      well_info_add_wells( well_info , ecl_file , report_nr , grid_nr );
    else
      well_info_add_UNRST_wells( well_info , ecl_file , grid_nr );
    
    ecl_file_close( ecl_file );
  } else
    util_abort("%s: invalid file type:%s - must be a restart file\n",__func__ , filename);
}


void well_info_free( well_info_type * well_info ) {
  hash_free( well_info->wells );
  stringlist_free( well_info->well_names );
  free( well_info );
}

int well_info_get_well_size( const well_info_type * well_info , const char * well_name ) {
  well_ts_type * well_ts = well_info_get_ts( well_info , well_name );
  return well_ts_get_size( well_ts );
}

/*****************************************************************/

well_state_type * well_info_get_state_from_time( const well_info_type * well_info , const char * well_name , time_t sim_time) {
  well_ts_type * well_ts = well_info_get_ts( well_info , well_name );
  return well_ts_get_state_from_sim_time( well_ts , sim_time );
}


well_state_type * well_info_get_state_from_report( const well_info_type * well_info , const char * well_name , int report_step ) {
  well_ts_type * well_ts = well_info_get_ts( well_info , well_name );
  return well_ts_get_state_from_report( well_ts , report_step);
}

well_state_type * well_info_iget_state( const well_info_type * well_info , const char * well_name , int time_index) {
  well_ts_type * well_ts = well_info_get_ts( well_info , well_name );
  return well_ts_iget_state( well_ts , time_index);
}

well_state_type * well_info_iiget_state( const well_info_type * well_info , int well_index , int time_index) {
  const char * well_name = stringlist_iget( well_info->well_names , well_index );
  return well_info_iget_state( well_info , well_name , time_index );
}

/*****************************************************************/

int well_info_get_num_wells( const well_info_type * well_info ) {
  return stringlist_get_size( well_info->well_names );
}

const char * well_info_iget_well_name( const well_info_type * well_info, int well_index) {
  return stringlist_iget( well_info->well_names , well_index);
}
