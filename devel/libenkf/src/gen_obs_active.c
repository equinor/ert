#include <stdlib.h>
#include <util.h>
#include <enkf_macros.h>
#include <gen_obs_active.h>


/**
   This file implements a small structure used to keep track of active
   and not active elements in a gen_obs object. It is used as argument
   when calling gen_obs_activate().
*/


#define GEN_OBS_ACTIVE_ID 67752

struct gen_obs_active_struct {
  int    __type_id;
  int    size;         /* The number of active cells. */
  int   *active_list;  /* A list of active indices. */
};


SAFE_CAST(gen_obs_active , GEN_OBS_ACTIVE_ID)



int gen_obs_active_get_active_size( const gen_obs_active_type * active) {
  return active->size;
}


int * gen_obs_active_alloc_list_copy(const gen_obs_active_type * active ) {
  return util_alloc_copy( active->active_list , active->size * sizeof * active->active_list , __func__);
}


const int * gen_obs_active_get_active_list(const gen_obs_active_type * active ) {
  return active->active_list;
}

