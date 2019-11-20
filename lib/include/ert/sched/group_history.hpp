/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'group_history.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_GROUP_HISTORY_H
#define ERT_GROUP_HISTORY_H

#ifdef __cplusplus
extern "C" {
#endif
#include <ert/util/time_t_vector.hpp>
#include <ert/util/stringlist.hpp>
#include <ert/util/util.hpp>

typedef struct group_history_struct group_history_type;


void                 group_history_free( group_history_type * group_history );
void                 group_history_fprintf( const group_history_type * group_history , int report_step , bool recursive , FILE * stream );


double               group_history_iget_GOPRH( const void * __group_history , int report_step );
double               group_history_iget_GGPRH( const void * __group_history , int report_step );
double               group_history_iget_GWPRH( const void * __group_history , int report_step );


UTIL_IS_INSTANCE_HEADER( group_history );



#ifdef __cplusplus
}
#endif
#endif
