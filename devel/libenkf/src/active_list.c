#include <stdlib.h>
#include <util.h>
#include <enkf_macros.h>
#include <active_list.h>
#include <int_vector.h>


/**
   This file implements a small structure used to denote which
   elements of a node/observation which is active. At the lowest level
   the active elements in a node is just a list of integers. This
   list of integers, with som extra twists is what is implemented
   here. 
   
   All the xxx_config objects have a pointer to an active_list
   instance. This pointer is passed to the enkf_serialize /
   enkf_deserialize routines.

   Observe that for the (very important!!) special case that all
   elements are active the (int *) pointer should not be accessed, and
   the code here is free to return NULL.


Example
-------

Consider a situation where faults number 0,4 and 5 should be active in
a fault object. Then the code will be like:


   ....
   active_list_reset(multflt_config->active_list);
   active_list_add_index(multflt_config->active_list , 0);
   active_list_add_index(multflt_config->active_list , 4);
   active_list_add_index(multflt_config->active_list , 5);
   ....

   When this fault object is serialized/deserialized only the elements
   0,4,5 are updated. 
*/


#define ACTIVE_LIST_TYPE_ID 66109

struct active_list_struct {
  int     	    __type_id;      /* Used for checking run_time casting. */
  int     	    active_size;    /* The number of active elements. */
  int     	    alloc_size;     /* The allocated size of the  active_list pointer. */
  active_mode_type  mode;           /* ALL_ACTIVE | INACTIVE | PARTLY_ACTIVE */
  int_vector_type  *index_list;     /* A list of active indices - if data_size == active_size this can be NULL. */
};

/*****************************************************************/


SAFE_CAST(active_list , ACTIVE_LIST_TYPE_ID)



/**
   The newly created active_list default to setting all indices actiove.
*/
active_list_type * active_list_alloc(active_mode_type mode) {
  active_list_type * active_list = util_malloc(sizeof * active_list , __func__);
  active_list->__type_id   = ACTIVE_LIST_TYPE_ID;
  active_list->index_list  = int_vector_alloc(0 , -1);
  active_list->mode        = mode;
  return active_list;
}


active_list_type * active_list_alloc_copy( const active_list_type * src) {
  active_list_type * new = active_list_alloc( ALL_ACTIVE );
  new->mode  = src->mode;
  int_vector_free( new->index_list ) ;
  new->index_list = int_vector_alloc_copy( src->index_list );
  return new;
}



void active_list_free( active_list_type * active_list ) {
  int_vector_free(active_list->index_list);
  free(active_list);
}



void active_list_free__( void * arg ) {
  active_list_type * active_list = active_list_safe_cast ( arg );
  active_list_free(active_list);
}





/*
  Setting the counter back to zero - i.e. a call to
  active_list_reset() will mean that we have *NO* active elements.
*/
void active_list_reset(active_list_type * active_list) {
  int_vector_reset( active_list->index_list );
}


/**
   Appends a new index to the current list of active indices, and
   setting the mode to PARTLY_ACTIVE.
*/
void active_list_add_index(active_list_type * active_list, int new_index) {
  active_list->mode = PARTLY_ACTIVE;
  int_vector_append( active_list->index_list , new_index );
}



/**
   Observe that this function should **ONLY** be called when mode ==
   PARTLY_ACTIVE, otherwise either all or none of the node's elements
   are active; and it is the responsability of the calling scope to
   infer that size.
*/

int active_list_get_active_size(const active_list_type * active_list) {
  if (active_list->mode == PARTLY_ACTIVE)
    return int_vector_size( active_list->index_list );
  else {
    util_abort("%s: internal error - asked for active size for object which is not PARTLY_ACTIVE \n");
    return 0;
  }
}


active_mode_type active_list_get_mode(const active_list_type * active_list) {
  return active_list->mode;
}



/**
   This will return a (const int *) pointer to the active indices. IFF
   (mode == INACTIVE || mode == ALL_ACTIVE) it will instead just
   return NULL. In that case it is the responsability of the calling
   scope to not dereference the NULL pointer.
*/

const int * active_list_get_active(const active_list_type * active_list) {
  if (active_list->mode == PARTLY_ACTIVE)
    return int_vector_get_const_ptr( active_list->index_list );
  else
    return NULL;  
}
