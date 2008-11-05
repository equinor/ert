#include <stdlib.h>
#include <util.h>
#include <enkf_macros.h>
#include <gen_data_active.h>


/**
   This file implements a small structure used to keep track of active
   and not active cells in a gen_data object. It is used as argument
   when calling gen_data_config_activate().
*/

#define GEN_DATA_ACTIVE_ID 61652

struct gen_data_active_struct {
  int    __type_id;
  int    size;         /* The number of active cells. */
  int   *active_list;  /* A list of active cell indices.*/
};


SAFE_CAST(gen_data_active , GEN_DATA_ACTIVE_ID);


int gen_data_active_get_active_size( const gen_data_active_type * active) {
  return active->size;
}


int * gen_data_active_alloc_list_copy(const gen_data_active_type * active ) {
  return util_alloc_copy( active->active_list , active->size * sizeof * active->active_list , __func__);
}

const int * gen_data_active_get_active_list(const gen_data_active_type * active ) {
  return active->active_list;
}

