#include <stdlib.h>
#include <util.h>
#include <stdbool.h>
#include <enkf_macros.h>
#include <field_active.h>


/**
   This file implements a small structure used to keep track of active
   and not active cells in a field. It is used as argument when
   calling field_config_activate(). 
*/

#define FIELD_ACTIVE_ID 66107

struct field_active_struct {
  int    __type_id;
  int    size;         /* The number of active cells. */
  int   *active_list;  /* A list of active cell indices - these should be in the adress space of the total
                          number of active cells in the currently active grid. */
};


SAFE_CAST(field_active , FIELD_ACTIVE_ID)



int field_active_get_active_size( const field_active_type * active) {
  return active->size;
}


int * field_active_alloc_list_copy(const field_active_type * active ) {
  return util_alloc_copy( active->active_list , active->size * sizeof * active->active_list , __func__);
}




