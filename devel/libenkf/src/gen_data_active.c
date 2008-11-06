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


void gen_data_active_update_active_list(const gen_data_active_type * gen_data_active , active_list_type * active_list) {
  int i;
  for (i=0; i < gen_data_active->size; i++)
    active_list_add_index(active_list , gen_data_active->active_list[i]);
}

