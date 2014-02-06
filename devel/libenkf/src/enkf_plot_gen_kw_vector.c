/*
   Copyright (C) 2014  Statoil ASA, Norway.

   The file 'enkf_plot_gen_kw_vector.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/util/stringlist.h>
#include <ert/util/thread_pool.h>
#include <ert/util/type_macros.h>
#include <ert/util/vector.h>

#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/enkf_node.h>
#include <ert/enkf/enkf_plot_gen_kw_vector.h>


#define ENKF_PLOT_GEN_KW_VECTOR_TYPE_ID 88362064

struct enkf_plot_gen_kw_vector_struct {
  UTIL_TYPE_ID_DECLARATION;
  int                           iens;
  double_vector_type          * data;
  const enkf_config_node_type * config_node;
  const stringlist_type       * key_list;
};


UTIL_IS_INSTANCE_FUNCTION( enkf_plot_gen_kw_vector , ENKF_PLOT_GEN_KW_VECTOR_TYPE_ID )


enkf_plot_gen_kw_vector_type * enkf_plot_gen_kw_vector_alloc( const enkf_config_node_type * config_node , int iens , const stringlist_type * key_list ) {
  enkf_plot_gen_kw_vector_type * vector = util_malloc( sizeof * vector );
  UTIL_TYPE_ID_INIT( vector , ENKF_PLOT_GEN_KW_VECTOR_TYPE_ID );
  vector->config_node = config_node;
  vector->data        = double_vector_alloc(0,0);
  vector->iens        = iens;
  vector->key_list    = key_list;
  return vector;
}


void enkf_plot_gen_kw_vector_free( enkf_plot_gen_kw_vector_type * vector ) {
  double_vector_free( vector->data );
  free( vector );
}


int enkf_plot_gen_kw_vector_get_size( const enkf_plot_gen_kw_vector_type * vector ) {
  return double_vector_size( vector->data );
}

double enkf_plot_gen_kw_vector_iget( const enkf_plot_gen_kw_vector_type * vector , int index )  {
  return double_vector_iget( vector->data , index );
}


void enkf_plot_gen_kw_vector_reset( enkf_plot_gen_kw_vector_type * vector ) {
  double_vector_reset( vector->data );
}


void enkf_plot_gen_kw_vector_load( enkf_plot_gen_kw_vector_type * vector , enkf_fs_type * fs , int report_step , state_enum state ) {
  enkf_plot_gen_kw_vector_reset( vector );
  {
    node_id_type node_id = { .report_step = report_step ,
                             .state       = state ,
                             .iens        = vector->iens };

    enkf_node_type * data_node = enkf_node_alloc( vector->config_node );
    int ikw;
    int n_kw = stringlist_get_size( vector->key_list );

    for (ikw = 0 ; ikw < n_kw ; ++ikw) {
      const char * key = stringlist_iget( vector->key_list , ikw );
      double value;

      if (enkf_node_user_get( data_node , fs , key , node_id , &value )) {
        double_vector_append(vector->data , value);
      }
    }

    enkf_node_free( data_node );
  }
}



void * enkf_plot_gen_kw_vector_load__( void * arg ) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  enkf_plot_gen_kw_vector_type * vector  = arg_pack_iget_ptr( arg_pack , 0);
  enkf_fs_type * fs = arg_pack_iget_ptr( arg_pack , 1 );
  int report_step = arg_pack_iget_int( arg_pack , 2 );
  state_enum state = arg_pack_iget_int( arg_pack , 3 );

  enkf_plot_gen_kw_vector_load( vector , fs , report_step , state );
  return NULL;
}
