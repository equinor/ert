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
#include <assert.h>
#include <stdbool.h>
#include <time.h>

#include <ert/util/bool_vector.h>
#include <ert/util/double_vector.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_config_node.h>
#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/enkf_plot_gen_kw.h>
#include <ert/enkf/enkf_plot_gen_kw_vector.h>
#include <ert/enkf/gen_kw_config.h>


#define ENKF_PLOT_GEN_KW_TYPE_ID 88362063

struct enkf_plot_gen_kw_struct {
  UTIL_TYPE_ID_DECLARATION;
  const enkf_config_node_type   * config_node;
  int                             size;        /* Number of ensembles. */
  enkf_plot_gen_kw_vector_type ** ensemble;    /* One vector for each ensemble. */
  stringlist_type               * key_list;    /* Name of individual keys for this GEN_KW. */
};


UTIL_IS_INSTANCE_FUNCTION( enkf_plot_gen_kw , ENKF_PLOT_GEN_KW_TYPE_ID )


enkf_plot_gen_kw_type * enkf_plot_gen_kw_alloc( const enkf_config_node_type * config_node ) {
  if (enkf_config_node_get_impl_type( config_node ) == GEN_KW) {
    enkf_plot_gen_kw_type * gen_kw = util_malloc( sizeof * gen_kw );
    UTIL_TYPE_ID_INIT( gen_kw , ENKF_PLOT_GEN_KW_TYPE_ID );
    gen_kw->config_node = config_node;
    gen_kw->size = 0;
    gen_kw->ensemble = NULL;
    gen_kw->key_list = stringlist_alloc_new();

    return gen_kw;
  }
  else {
    return NULL;
  }
}


void enkf_plot_gen_kw_free( enkf_plot_gen_kw_type * gen_kw ) {
  int iens;
  for (iens = 0 ; iens < gen_kw->size ; ++iens) {
    enkf_plot_gen_kw_vector_free( gen_kw->ensemble[iens] );
  }
  stringlist_free( gen_kw->key_list );
  free( gen_kw );
}


int enkf_plot_gen_kw_get_size( const enkf_plot_gen_kw_type * gen_kw ) {
  return gen_kw->size;
}

enkf_plot_gen_kw_vector_type * enkf_plot_gen_kw_iget( const enkf_plot_gen_kw_type * gen_kw , int iens)  {
  assert(iens >= 0 && iens < gen_kw->size);
  return gen_kw->ensemble[iens];
}


static void enkf_plot_gen_kw_resize( enkf_plot_gen_kw_type * gen_kw , int new_size ) {
  if (new_size != gen_kw->size) {
    int iens;

    if (new_size < gen_kw->size) {
      for (iens = new_size; iens < gen_kw->size; iens++) {
        enkf_plot_gen_kw_vector_free( gen_kw->ensemble[iens] );
      }
    }

    gen_kw->ensemble = util_realloc( gen_kw->ensemble , new_size * sizeof * gen_kw->ensemble);

    if (new_size > gen_kw->size) {
      for (iens = gen_kw->size; iens < new_size; iens++) {
        gen_kw->ensemble[iens] = enkf_plot_gen_kw_vector_alloc( gen_kw->config_node , iens , gen_kw->key_list );
      }
    }
    gen_kw->size = new_size;
  }
}


static void enkf_plot_gen_kw_reset( enkf_plot_gen_kw_type * gen_kw ) {
  const gen_kw_config_type * gen_kw_config = enkf_config_node_get_ref( gen_kw->config_node );

  stringlist_free( gen_kw->key_list );
  gen_kw->key_list = gen_kw_config_alloc_name_list( gen_kw_config );
}


const stringlist_type * enkf_plot_gen_kw_get_keys( const enkf_plot_gen_kw_type * gen_kw ) {
  return gen_kw->key_list;
}


void enkf_plot_gen_kw_load( enkf_plot_gen_kw_type  * plot_gen_kw,
                            enkf_fs_type           * fs,
                            int                      report_step,
                            state_enum               state,
                            const bool_vector_type * input_mask ) {

  state_map_type * state_map = enkf_fs_get_state_map( fs );
  int ens_size = state_map_get_size( state_map );
  bool_vector_type * mask;

  if (input_mask)
    mask = bool_vector_alloc_copy( input_mask );
  else
    mask = bool_vector_alloc( ens_size , true );

  enkf_plot_gen_kw_reset( plot_gen_kw );
  enkf_plot_gen_kw_resize( plot_gen_kw , ens_size );

  {
    int iens;
    enkf_node_type * data_node;

    data_node = enkf_node_alloc( plot_gen_kw->config_node );

    for (iens = 0; iens < ens_size; ++iens) {
      if (bool_vector_iget( mask , iens)) {
        enkf_plot_gen_kw_vector_type * vector = enkf_plot_gen_kw_iget( plot_gen_kw , iens );
        enkf_plot_gen_kw_vector_load( vector , fs , report_step , state );
      }
    }

    enkf_node_free( data_node );
  }
}
