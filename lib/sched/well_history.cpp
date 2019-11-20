/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'well_history.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <string.h>

#include <ert/util/double_vector.hpp>
#include <ert/util/int_vector.hpp>
#include <ert/util/time_t_vector.hpp>
#include <ert/util/size_t_vector.hpp>
#include <ert/util/bool_vector.hpp>

#include <ert/sched/sched_types.hpp>
#include <ert/sched/well_history.hpp>
#include <ert/sched/group_history.hpp>
#include <ert/sched/sched_kw.hpp>
#include <ert/sched/sched_kw_wconhist.hpp>
#include <ert/sched/sched_kw_wconinje.hpp>
#include <ert/sched/sched_kw_wconinjh.hpp>


#define WELL_HISTORY_TYPE_ID   44431872


 /**
    The following paradigm applies to the schedule files:

      o It is based on keywords which will apply until the next
        occurence of the same keyword.

      o It is naturaly organized in blocks delimited by DATES / TSTEP
        keywords.


       Example
       -------

       DATES
          1  'JAN'  2000 /

       WCONHIST
           'OP_1'      'OPEN'      'ORAT'   7996.000      4.000 1.46402E+006  5* /
       /

       DATES
          1  'FEB'  2000 /

                                    <--- What is the oil rate here?

       DATES
          1  'MAR'  2000 /


       END


       In this example an historical oil production rate of 7996 is
       specified for immediately following the DATES keyword at
       1. 'JAN' - but what is the oil production rate at 15.th of
       February? The point is that the 7996 rate applies until it is
       set again, i.e. the DATES / TSTEP are not that suitable as block
       units.


    To support this behaviour the vector types from vector_template.c
    are well suited; setting the default with
    e.g. double_vector_set_default() will ascertain that the default
    value is returned for all subsequent gets from the vector beyond
    the set length. (That was clear ....)

*/



struct well_history_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                * well_name;
  int_vector_type     * kw_type;         /* This enum should be one of: NONE(default), WCONHIST , WCONINJE and WCONINJH (sched_kw_type_enum in sched_types.h). */
  size_t_vector_type  * active_state;    /* Contains pointer to the currently active of the xxx_state objects. The size_t_vector instance is abused to store pointer values (i.e. addresses). */
  wconhist_state_type * wconhist_state;
  wconinje_state_type * wconinje_state;
  wconinjh_state_type * wconinjh_state;
  size_t_vector_type  * parent;
  bool_vector_type    * well_open;
};



UTIL_SAFE_CAST_FUNCTION( well_history , WELL_HISTORY_TYPE_ID )
static UTIL_SAFE_CAST_FUNCTION_CONST( well_history , WELL_HISTORY_TYPE_ID )
UTIL_IS_INSTANCE_FUNCTION( well_history , WELL_HISTORY_TYPE_ID)

void well_history_set_parent( well_history_type * child_well , int report_step , const group_history_type * parent_group) {
  size_t_vector_iset_default( child_well->parent , report_step , ( size_t ) parent_group);
}

group_history_type * well_history_get_parent( well_history_type * child_well , int report_step ) {
  return (group_history_type *) size_t_vector_safe_iget( child_well->parent , report_step );
}

void well_history_free( well_history_type * well_history ) {
   free( well_history->well_name );
   int_vector_free( well_history->kw_type );
   wconhist_state_free( well_history->wconhist_state );
   wconinjh_state_free( well_history->wconinjh_state );
   wconinje_state_free( well_history->wconinje_state );
   size_t_vector_free( well_history->active_state );
   bool_vector_free( well_history->well_open );
   size_t_vector_free( well_history->parent );
   free( well_history );
}


void well_history_free__( void * arg ) {
  well_history_free( well_history_safe_cast( arg ));
}


/*****************************************************************/

double well_history_iget_WOPRH( const well_history_type * well_history , int report_step ) {
  sched_kw_type_enum current_type = (sched_kw_type_enum)int_vector_safe_iget( well_history->kw_type , report_step );
  if (current_type == WCONHIST) {
    void * state_ptr = (void *) size_t_vector_safe_iget( well_history->active_state , report_step );
    return wconhist_state_iget_WOPRH( state_ptr , report_step );
  } else
    return 0;  /* If it is not in WCONHIST state we return 0 with no further ado. */
}


double well_history_iget_WWPRH( const well_history_type * well_history , int report_step ) {
  sched_kw_type_enum current_type = (sched_kw_type_enum)int_vector_safe_iget( well_history->kw_type , report_step );
  if (current_type == WCONHIST) {
    void * state_ptr = (void *) size_t_vector_safe_iget( well_history->active_state , report_step );
    return wconhist_state_iget_WWPRH( state_ptr , report_step );
  } else
    return 0;  /* If it is not in WCONHIST state we return 0 with no further ado. */
}


double well_history_iget_WGPRH( const well_history_type * well_history , int report_step ) {
  sched_kw_type_enum current_type = (sched_kw_type_enum)int_vector_safe_iget( well_history->kw_type , report_step );
  if (current_type == WCONHIST) {
    void * state_ptr = (void *) size_t_vector_safe_iget( well_history->active_state , report_step );
    return wconhist_state_iget_WGPRH( state_ptr , report_step );
  } else
    return 0;  /* If it is not in WCONHIST state we return 0 with no further ado. */
}
