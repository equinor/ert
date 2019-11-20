/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'group_index.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/size_t_vector.hpp>
#include <ert/util/int_vector.hpp>
#include <ert/util/util.hpp>

#include <ert/sched/sched_types.hpp>
#include <ert/sched/group_index.hpp>


#define GROUP_INDEX_TYPE_ID 96580631


struct group_index_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                         * group_name;
  char                         * variable;
  const void                   * group_history;
  sched_history_callback_ftype * func;
};





UTIL_IS_INSTANCE_FUNCTION( group_index , GROUP_INDEX_TYPE_ID )
UTIL_SAFE_CAST_FUNCTION_CONST( group_index , GROUP_INDEX_TYPE_ID )


void group_index_free( group_index_type * index ) {
  free( index->group_name );
  free( index->variable );
  free( index );
}


sched_history_callback_ftype * group_index_get_callback( const group_index_type * group_index ) {
  return group_index->func;
}


const void * group_index_get_state( const group_index_type * group_index ) {
  return group_index->group_history;
}



const void * group_index_get_state__( const void * index ) {
  const group_index_type * group_index = group_index_safe_cast_const( index );
  return group_index_get_state( group_index );
}
