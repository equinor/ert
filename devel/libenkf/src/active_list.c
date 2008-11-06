#include <stdlib.h>
#include <util.h>
#include <enkf_macros.h>
#include <active_list.h>


/**
   This file implements a small simple structure used to denote which
   elements of a node/observation which is active. At the lowest level
   the elements in a node are just in a list of integers. This list of
   integers, with som extra twists is what is implemneted here. 

   Observe that for the special case that all elements are active the
   (int *) pointer should not be accessed, and the code here is free
   to return NULL.
*/


#define ACTIVE_LIST_TYPE_ID 66109

struct active_list_struct {
  int     __type_id;
  int     active_size;    /* The number of active elements. */
  int     data_size;      /* The data size of the node in question. */
  int     alloc_size;     /* The allocated size of the  active_list pointer. */
  int    *active_list;    /* A list of active indices - if data_size == active_size this can be NULL. */
};

/*****************************************************************/

SAFE_CAST(active_list , ACTIVE_LIST_TYPE_ID)


static void active_list_realloc_list(active_list_type * active_list , int new_alloc_size) {
  active_list->alloc_size = new_alloc_size;
  active_list->active_list = util_realloc(active_list->active_list , new_alloc_size * sizeof * active_list->active_list , __func__);
}



/**
   The newly created active_list default to setting all indices actiove.
*/
active_list_type * active_list_alloc(int data_size) {
  active_list_type * active_list = util_malloc(sizeof * active_list , __func__);
  active_list->__type_id   = ACTIVE_LIST_TYPE_ID;
  active_list->active_size = data_size;    
  active_list->data_size   = data_size;
  active_list->active_list = NULL;
  active_list_set_all_active(active_list);
  return active_list;
}


void active_list_free( active_list_type * active_list ) {
  util_safe_free( active_list->active_list );
  free(active_list);
}


void active_list_set_all_active(active_list_type * active_list) {
  active_list->active_size = active_list->data_size;
  active_list_realloc_list(active_list , 0); /* All active - do not need the (int *) storage. */
}


void active_list_set_data_size(active_list_type * active_list , int data_size) {
  /* Iff we all_active allready, then we assume that also applies in
     the new situation. */

  if (active_list->active_size == active_list->data_size)
    active_list->active_size = data_size;
  active_list->data_size = data_size;
}


void active_list_grow(active_list_type * active_list , int delta) {
  active_list_set_data_size( active_list , delta + active_list->data_size );
}


void active_list_reset(active_list_type * active_list) {
  active_list->active_size = 0;
  if (active_list->alloc_size > 100)
    active_list_realloc_list(active_list , 100);
}


/**
   Appends a new index to the current list of active indices. 
*/
void active_list_add_index(active_list_type * active_list, int new_index) {
  if (active_list->active_size == active_list->data_size)
    util_abort("%s: all elements are already active - can not add more ..\n",__func__);

  if (active_list->active_size == active_list->alloc_size)
    active_list_realloc_list(active_list , 2 + 2*active_list->alloc_size);
  
  active_list->active_list[active_list->active_size] = new_index;
  active_list->active_size++;
}


int active_list_get_active_size(const active_list_type * active_list) {
  return active_list->active_size;
}


const int  * active_list_get_active(const active_list_type * active_list) {
  if (active_list->active_size  < active_list->data_size)
    return active_list->active_list;
  else
    return NULL;
}
