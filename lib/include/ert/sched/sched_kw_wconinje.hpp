/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'sched_kw_wconinje.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_SCHED_KW_WCONINJE_H
#define ERT_SCHED_KW_WCONINJE_H


#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>

#include <ert/util/stringlist.hpp>
#include <ert/util/buffer.hpp>
#include <ert/util/time_t_vector.hpp>

#include <ert/sched/sched_macros.hpp>
#include <ert/sched/sched_types.hpp>


typedef struct sched_kw_wconinje_struct sched_kw_wconinje_type;
typedef struct wconinje_state_struct    wconinje_state_type;


bool                     sched_kw_wconinje_well_open( const sched_kw_wconinje_type * kw, const char * well_name);
char **                  sched_kw_wconinje_alloc_wells_copy( const sched_kw_wconinje_type * , int * );

double                   sched_kw_wconinje_get_surface_flow( const sched_kw_wconinje_type * kw , const char * well);
bool                     sched_kw_wconinje_has_well( const sched_kw_wconinje_type * , const char * );
bool                     sched_kw_wconinje_buffer_fwrite( const sched_kw_wconinje_type * kw , const char * well_name , buffer_type * buffer);

void                     sched_kw_wconinje_close_state(wconinje_state_type * state , int report_step );
void                     sched_kw_wconinje_update_state( const sched_kw_wconinje_type * kw , wconinje_state_type * state , const char * well_name , int report_step );
wconinje_state_type    * wconinje_state_alloc( const char * well_name , const time_t_vector_type * time);
void                     wconinje_state_free( wconinje_state_type * wconinje );

/*******************************************************************/



KW_HEADER(wconinje)



#ifdef __cplusplus
}
#endif
#endif
