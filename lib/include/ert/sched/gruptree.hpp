/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'gruptree.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_GRUPTREE_H
#define ERT_GRUPTREE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

typedef struct gruptree_struct gruptree_type;

gruptree_type * gruptree_alloc();

void            gruptree_register_grup(gruptree_type *, const char *, const char *);
void            gruptree_register_well(gruptree_type *, const char *, const char *);
char         ** gruptree_alloc_grup_well_list(gruptree_type *, const char *, int *);

#ifdef __cplusplus
}
#endif
#endif
