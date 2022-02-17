/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'ranking_table.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_RANKING_TABLE_H
#define ERT_RANKING_TABLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include <ert/util/perm_vector.h>

#include <ert/enkf/misfit_ensemble_typedef.hpp>

typedef struct ranking_table_struct ranking_table_type;

void ranking_table_set_ens_size(ranking_table_type *table, int ens_size);
ranking_table_type *ranking_table_alloc(int ens_size);
void ranking_table_free(ranking_table_type *table);

bool ranking_table_fwrite_ranking(const ranking_table_type *ranking_table,
                                  const char *ranking_key,
                                  const char *filename);

const perm_vector_type *
ranking_table_get_permutation(const ranking_table_type *ranking_table,
                              const char *ranking_key);

#ifdef __cplusplus
}
#endif

#endif
