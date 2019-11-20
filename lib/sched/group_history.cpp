/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'group_history.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdbool.h>

#include <ert/util/util.hpp>
#include <ert/util/size_t_vector.hpp>
#include <ert/util/time_t_vector.hpp>
#include <ert/util/vector.hpp>

#include <ert/sched/group_history.hpp>
#include <ert/sched/well_history.hpp>
#include <ert/sched/group_index.hpp>

#define GROUP_HISTORY_TYPE_ID 5100635


struct group_history_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                     * group_name;
  bool                       well_group;       /* If true this group contains wells, otehrwise it contains other groups. */
  const time_t_vector_type * time;
  int                        start_time;       /* At which report step was this group first defined? */
  size_t_vector_type       * parent;
  size_t_vector_type       * children;
  vector_type              * children_storage;
  int                        __active_step;    /* Internal variable to ensure that several repeated calls to add_child / del_child work on the correct child_hash instance. */
};

UTIL_SAFE_CAST_FUNCTION( group_history , GROUP_HISTORY_TYPE_ID )
UTIL_SAFE_CAST_FUNCTION_CONST( group_history , GROUP_HISTORY_TYPE_ID )
UTIL_IS_INSTANCE_FUNCTION( group_history , GROUP_HISTORY_TYPE_ID)


void group_history_free( group_history_type * group_history ) {
  vector_free( group_history->children_storage );
  size_t_vector_free( group_history->children );
  size_t_vector_free( group_history->parent );

  free( group_history->group_name );
  free( group_history );
}


static void group_history_set_parent( group_history_type * child_group , int report_step , const group_history_type * parent_group) {
  size_t_vector_iset_default( child_group->parent , report_step , ( size_t ) parent_group);
}


static group_history_type * group_history_get_parent( group_history_type * child_group , int report_step ) {
  return (group_history_type *) size_t_vector_safe_iget( child_group->parent , report_step );
}



static void group_history_ensure_private_child_list(group_history_type * group_history , int report_step) {
  if (group_history->__active_step != report_step) {
    /*
      We wish to modify the child-parent relationships at this
      report_step; and the internal flag __active_step tells us that
      the current elemement in the children pointer table is not
      created at this report step; i.e. we must create a new child
      hash instance and install at this report step.

      For each repeated call the input parameter @report_step must
      be >= @report_step from the previous call; otherwise things
      will break hard.
    */

    hash_type * new_child_hash = hash_alloc();                                                     /* Allocate the new new hash table,
                                                                                                      and fill it up with the current list
                                                                                                      of children. */
    {
      hash_type * old_child_hash = ( hash_type * ) size_t_vector_safe_iget( group_history->children , report_step );
      hash_iter_type * old_iter = hash_iter_alloc( old_child_hash );
      while (!hash_iter_is_complete( old_iter )) {
        const char * child_name = hash_iter_get_next_key( old_iter );
        void * child            = hash_get( old_child_hash , child_name );

        hash_insert_ref( new_child_hash , child_name , child );
      }
      hash_iter_free( old_iter );
    }

    vector_append_owned_ref( group_history->children_storage , new_child_hash , hash_free__ );        /* Store it in the storge area. */
    size_t_vector_iset_default( group_history->children , report_step , ( size_t ) new_child_hash );
    group_history->__active_step = report_step;
  }
}


static void group_history_del_child( group_history_type * group_history , const char * child_name , int report_step ) {
  group_history_ensure_private_child_list( group_history , report_step );
  {
    hash_type * child_hash = (hash_type *) size_t_vector_iget( group_history->children , report_step );
    hash_del( child_hash , child_name );
  }
}


void group_history_fprintf(const group_history_type * group_history , int report_step , bool recursive , FILE * stream ) {
  fprintf(stream , "\n----------------------------------------------------------------------\n");
  fprintf(stream , "Group: %s \n",group_history->group_name);

  {
    hash_type * child_hash = (hash_type *) size_t_vector_safe_iget( group_history->children , report_step );
    hash_iter_type * child_iter = hash_iter_alloc( child_hash );
    int   counter = 0;
    while (!hash_iter_is_complete( child_iter )) {

      const char * name  = hash_iter_get_next_key( child_iter );
      const void * child = hash_get( child_hash , name );
      if ( group_history_is_instance( child ))
        fprintf(stream , "%8s(G) ",name);
      else
        fprintf(stream , "%8s(W) ",name);
      counter++;
      if ((counter % 4) == 0)
        fprintf(stream , "\n");

    }
    fprintf(stream , "\n----------------------------------------------------------------------\n");
    if (recursive) {
      hash_iter_restart( child_iter );
      while (!hash_iter_is_complete( child_iter )) {

        const char * name  = hash_iter_get_next_key( child_iter );
        const void * child = hash_get( child_hash , name );
        if ( group_history_is_instance( child ))
          group_history_fprintf( (const group_history_type*)child , report_step , recursive , stream );
      }
    }
    hash_iter_free( child_iter );
  }
}


/*****************************************************************/

double group_history_iget_GOPRH( const void * __group_history , int report_step ) {
  const group_history_type * group_history = group_history_safe_cast_const( __group_history );
  {
    double GOPRH = 0;

    hash_type      * child_hash = (hash_type *) size_t_vector_safe_iget( group_history->children , report_step );   /* Get a pointer to child hash instance valid at this report_step. */
    hash_iter_type * child_iter = hash_iter_alloc( child_hash );
    while ( !hash_iter_is_complete( child_iter )) {
      const char * child_name = hash_iter_get_next_key( child_iter );
      const void * child      = hash_get( child_hash , child_name );
      if (group_history_is_instance( child ))
        GOPRH += group_history_iget_GOPRH( child , report_step );
      else {
        double WOPRH = well_history_iget_WOPRH( (const well_history_type*)child , report_step );
        GOPRH += WOPRH;
      }
    }
    hash_iter_free( child_iter );

    return GOPRH;
  }
}



double group_history_iget_GWPRH( const void * __group_history , int report_step ) {
  const group_history_type * group_history = group_history_safe_cast_const( __group_history );
  {
    double GWPRH = 0;

    hash_type      * child_hash = (hash_type *) size_t_vector_safe_iget( group_history->children , report_step );   /* Get a pointer to child hash instance valid at this report_step. */
    hash_iter_type * child_iter = hash_iter_alloc( child_hash );
    while ( !hash_iter_is_complete( child_iter )) {
      const char * child_name = hash_iter_get_next_key( child_iter );
      const void * child      = hash_get( child_hash , child_name );
      if (group_history_is_instance( child ))
        GWPRH += group_history_iget_GWPRH( child , report_step );
      else
        GWPRH += well_history_iget_WWPRH( (const well_history_type*)child , report_step );
    }
    hash_iter_free( child_iter );

    return GWPRH;
  }
}




double group_history_iget_GGPRH( const void * __group_history , int report_step ) {
  const group_history_type * group_history = group_history_safe_cast_const( __group_history );
  {
    double GGPRH = 0;

    hash_type      * child_hash = (hash_type *) size_t_vector_safe_iget( group_history->children , report_step );   /* Get a pointer to child hash instance valid at this report_step. */
    hash_iter_type * child_iter = hash_iter_alloc( child_hash );
    while ( !hash_iter_is_complete( child_iter )) {
      const char * child_name = hash_iter_get_next_key( child_iter );
      const void * child      = hash_get( child_hash , child_name );
      if (group_history_is_instance( child ))
        GGPRH += group_history_iget_GGPRH( child , report_step );
      else
        GGPRH += well_history_iget_WGPRH( (const well_history_type*)child , report_step );
    }
    hash_iter_free( child_iter );

    return GGPRH;
  }
}
