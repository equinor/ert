/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'group_index.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_GROUP_INDEX_H
#define ERT_GROUP_INDEX_H

#ifdef __cplusplus
extern "C" {
#endif
#include <ert/util/type_macros.hpp>

#include <ert/sched/sched_types.hpp>

typedef struct group_index_struct group_index_type;

void                            group_index_free( group_index_type * group_index );
sched_history_callback_ftype *  group_index_get_callback( const group_index_type * group_index );
const void                   *  group_index_get_state__( const void * index );
const void                   *  group_index_get_state( const group_index_type * group_index );



UTIL_IS_INSTANCE_HEADER( group_index );
UTIL_SAFE_CAST_HEADER_CONST( group_index );

#ifdef __cplusplus
}
#endif
#endif
