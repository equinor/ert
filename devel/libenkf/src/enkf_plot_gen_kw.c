/*
   Copyright (C) 2014  Statoil ASA, Norway.

   The file 'enkf_plot_blockvector.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/bool_vector.h>
#include <ert/util/double_vector.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_config_node.h>
#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/enkf_plot_gen_kw.h>


#define ENKF_PLOT_GEN_KW_TYPE_ID 88362063

struct enkf_plot_gen_kw_struct {
  UTIL_TYPE_ID_DECLARATION;
  const enkf_config_node_type * config_node;
  int                           size;
  double_vector_type          * data;
/*
  int                           n_keys;
  const enkf_config_node_type * config_node;
  const stringlist_type       * keys;  / * Name of individual keys for this GEN_KW. * /
  double_vector_type         ** data;  / * Data-vectors, one for each key. * /
  arg_pack_type              ** work_arg;
  int                           step;
  state_enum                    state;
*/
};


UTIL_IS_INSTANCE_FUNCTION( enkf_plot_gen_kw , ENKF_PLOT_GEN_KW_TYPE_ID )


enkf_plot_gen_kw_type * enkf_plot_gen_kw_alloc( const enkf_config_node_type * config_node) {
  if (enkf_config_node_get_impl_type( config_node ) == GEN_KW) {
    enkf_plot_gen_kw_type * gen_kw = util_malloc( sizeof * gen_kw );
    UTIL_TYPE_ID_INIT( gen_kw , ENKF_PLOT_GEN_KW_TYPE_ID );
    gen_kw->config_node = config_node;
    gen_kw->size = 0;
    gen_kw->data = double_vector_alloc( 0 , 0.0 );

/*
    data->keys = NULL;
    data->n_keys = 0;
    data->work_arg = NULL;
    data->step = -1;
    data->state = UNDEFINED;
*/

    return gen_kw;
  }
  else {
    return NULL;
  }
}


void enkf_plot_gen_kw_free( enkf_plot_gen_kw_type * gen_kw ) {
  double_vector_free( gen_kw->data );
  free( gen_kw );
}


int enkf_plot_gen_kw_get_size( const enkf_plot_gen_kw_type * gen_kw ) {
  return double_vector_size( gen_kw->data );
}

double enkf_plot_gen_kw_iget( const enkf_plot_gen_kw_type * gen_kw , int iens)  {
  return double_vector_iget( gen_kw->data , iens );
}


void enkf_plot_gen_kw_reset( enkf_plot_gen_kw_type * gen_kw ) {
  double_vector_reset( gen_kw->data );
  gen_kw->size = 0;
}


void enkf_plot_gen_kw_load( enkf_plot_gen_kw_type  * plot_gen_kw,
                            enkf_fs_type           * fs,
                            int                      report_step,
                            state_enum               state,
                            const char             * key,
                            const bool_vector_type * input_mask) {
  enkf_plot_gen_kw_reset( plot_gen_kw );

  state_map_type * state_map = enkf_fs_get_state_map( fs );
  int ens_size = state_map_get_size( state_map );
  bool_vector_type * mask;

  if (input_mask)
    mask = bool_vector_alloc_copy( input_mask );
  else
    mask = bool_vector_alloc( ens_size , false );

  state_map_select_matching( state_map , mask , STATE_HAS_DATA );

  plot_gen_kw->size = ens_size;

  {
    int iens;
    enkf_node_type * data_node;

    if (enkf_config_node_get_impl_type( plot_gen_kw->config_node ) == CONTAINER)
      data_node = enkf_node_alloc_private_container( plot_gen_kw->config_node );
    else
      data_node = enkf_node_alloc( plot_gen_kw->config_node );

    for (iens = 0; iens < ens_size; ++iens) {
      if (bool_vector_iget( mask , iens)) {
        double value;
        node_id_type node_id = { .report_step = report_step ,
                                 .state       = state ,
                                 .iens        = iens };

        if (enkf_node_user_get( data_node , fs , key , node_id , &value)) {
          double_vector_append(plot_gen_kw->data , value);
        }
      }
    }

    enkf_node_free( data_node );
  }
}
