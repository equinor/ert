/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'sched_kw_wconhist.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_SCHED_KW_WCONHIST_H
#define ERT_SCHED_KW_WCONHIST_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>

#include <ert/util/type_macros.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/stringlist.hpp>
#include <ert/util/time_t_vector.hpp>

#include <ert/sched/sched_types.hpp>
#include <ert/sched/sched_macros.hpp>

typedef  struct  sched_kw_wconhist_struct sched_kw_wconhist_type;
typedef  struct  wconhist_state_struct    wconhist_state_type;

#define WCONHIST_DEFAULT_STATUS  OPEN

sched_kw_wconhist_type * sched_kw_wconhist_fscanf_alloc( FILE *, bool *, const char *);
void                     sched_kw_wconhist_free(sched_kw_wconhist_type * );
void                     sched_kw_wconhist_fprintf(const sched_kw_wconhist_type * , FILE *);
void                     sched_kw_wconhist_fwrite(const sched_kw_wconhist_type *, FILE *);
sched_kw_wconhist_type * sched_kw_wconhist_fread_alloc( FILE *);
hash_type              * sched_kw_wconhist_alloc_well_obs_hash(const sched_kw_wconhist_type *);
double                   sched_kw_wconhist_get_orat( sched_kw_wconhist_type * kw , const char * well_name);
void                     sched_kw_wconhist_set_surface_flow(  sched_kw_wconhist_type * kw , const char * well_name , double orat);
bool                     sched_kw_wconhist_has_well( const sched_kw_wconhist_type * kw , const char * well_name);
bool                     sched_kw_wconhist_well_open( const sched_kw_wconhist_type * kw, const char * well_name);
void                     sched_kw_wconhist_update_state(const sched_kw_wconhist_type * kw , wconhist_state_type * state , const char * well_name , int report_step );


wconhist_state_type    * wconhist_state_alloc( const time_t_vector_type * time);
void                     wconhist_state_free( wconhist_state_type * wconhist );

double                   wconhist_state_iget_WOPRH( const void * state , int report_step );
double                   wconhist_state_iget_WGPRH( const void * state , int report_step );
double                   wconhist_state_iget_WWPRH( const void * state , int report_step );

void                     sched_kw_wconhist_close_state(wconhist_state_type * state , int report_step );


UTIL_SAFE_CAST_HEADER( sched_kw_wconhist );
/*******************************************************************/



KW_HEADER(wconhist)

#ifdef __cplusplus
}
#endif
#endif
