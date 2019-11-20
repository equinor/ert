/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'sched_kw.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_SCHED_KW_H
#define ERT_SCHED_KW_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include <ert/util/hash.hpp>

#include <ert/sched/sched_types.hpp>

typedef struct sched_kw_struct sched_kw_type;



/*****************************************************************/



  void sched_kw_free__(void *);
  sched_kw_type_enum      sched_kw_get_type(const sched_kw_type *);
  sched_kw_type         * sched_kw_token_alloc(const stringlist_type * tokens, int * token_index, hash_type * fixed_length_table, bool * foundEND);
  void                    sched_kw_fprintf(const sched_kw_type *, FILE *);
  void                    sched_kw_free(sched_kw_type *);

  sched_kw_type        ** sched_kw_split_alloc_DATES(const sched_kw_type *, int *);
  time_t                  sched_kw_get_new_time(const sched_kw_type *, time_t);
  void                  * sched_kw_get_data( sched_kw_type * kw);
  const void            * sched_kw_get_const_data( const sched_kw_type * kw);
  const char            * sched_kw_get_name( const sched_kw_type * kw);
  bool                    sched_kw_has_well( const sched_kw_type * sched_kw , const char * well );
  bool                    sched_kw_well_open( const sched_kw_type * sched_kw , const char * well );

#ifdef __cplusplus
}
#endif
#endif
