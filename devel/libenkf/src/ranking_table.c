/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'ranking_table.c' is part of ERT - Ensemble based Reservoir Tool. 
   
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
#include <stdio.h>
#include <math.h>

#include <util.h>
#include <hash.h>
#include <vector.h>
#include <double_vector.h>
#include <msg.h>
#include <buffer.h>
#include <type_macros.h>

#include <enkf_obs.h>
#include <enkf_fs.h>
#include <enkf_util.h>
#include <ranking_common.h>
#include <data_ranking.h>
#include <misfit_ranking.h>
#include <misfit_table.h>
#include <ranking_table.h>


#define RANKING_TABLE_TYPE_ID 78420651

struct ranking_table_struct {
  UTIL_TYPE_ID_DECLARATION;
  int                 ens_size;    // Will not really handle ensemble resize events
  hash_type         * ranking_table;
};



void ranking_table_free( ranking_table_type * table ) {
  hash_free( table->ranking_table );
  free( table );
}



void ranking_table_set_ens_size( ranking_table_type * table, int ens_size) {
  table->ens_size = ens_size;
}

ranking_table_type * ranking_table_alloc( int ens_size ) {
  ranking_table_type * table = util_malloc( sizeof * table );
  table->ranking_table = hash_alloc();
  return table;
}


void ranking_table_add_data_ranking( ranking_table_type * ranking_table , const char * ranking_key , const char * user_key , const char * key_index , 
                                     enkf_fs_type * fs , enkf_config_node_type * config_node , int step , state_enum state) {
  
  data_ranking_type * ranking = data_ranking_alloc( ranking_table->ens_size , user_key , key_index , fs , config_node , step , state );
  hash_insert_hash_owned_ref( ranking_table->ranking_table , ranking_key , ranking, data_ranking_free__ );

}


bool ranking_table_has_ranking( const ranking_table_type * ranking_table , const char * ranking_key ) {
  return hash_has_key( ranking_table->ranking_table , ranking_key );
}



bool ranking_table_display_ranking( const ranking_table_type * ranking_table , const char * ranking_key ) {
  if (hash_has_key( ranking_table->ranking_table , ranking_key)) {
    void * ranking = hash_get( ranking_table->ranking_table , ranking_key );
    
    if (data_ranking_is_instance( ranking )) {
      data_ranking_type * data_ranking = data_ranking_safe_cast( ranking );
      data_ranking_display( data_ranking );
    }
    

    return true;
  } else
    return false;
}
