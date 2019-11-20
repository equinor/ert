/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'sched_kw_gruptree.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_SCHED_KW_GRUPTREE_H
#define ERT_SCHED_KW_GRUPTREE_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>

#include <ert/util/stringlist.hpp>

#include <ert/sched/sched_macros.hpp>

typedef struct sched_kw_gruptree_struct sched_kw_gruptree_type;

void                     sched_kw_gruptree_alloc_child_parent_list(const sched_kw_gruptree_type *, char ***, char ***, int *);

/*******************************************************************/



KW_HEADER(gruptree)


#ifdef __cplusplus
}
#endif
#endif
