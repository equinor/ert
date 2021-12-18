/*
   Copyright (C) 2019  Equinor ASA, Norway.

   The file 'ies_enkf_config.hpp' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef IES_ENKF_CONFIG_H
#define IES_ENKF_CONFIG_H

#include <stdbool.h>

namespace ies {

typedef enum {
    IES_INVERSION_EXACT = 0,
    IES_INVERSION_SUBSPACE_EXACT_R = 1,
    IES_INVERSION_SUBSPACE_EE_R = 2,
    IES_INVERSION_SUBSPACE_RE = 3
} inversion_type;

typedef struct enkf_config_struct enkf_config_type;

enkf_config_type *enkf_config_alloc();
void enkf_config_free(enkf_config_type *config);

int enkf_config_get_enkf_subspace_dimension(const enkf_config_type *config);
void enkf_config_set_enkf_subspace_dimension(enkf_config_type *config,
                                             int subspace_dimension);

double enkf_config_get_truncation(const enkf_config_type *config);
void enkf_config_set_truncation(enkf_config_type *config, double truncation);

void enkf_config_set_option_flags(enkf_config_type *config, long flags);
long enkf_config_get_option_flags(const enkf_config_type *config);

double enkf_config_get_max_steplength(const enkf_config_type *config);
void enkf_config_set_max_steplength(enkf_config_type *config,
                                    double max_steplength);

double enkf_config_get_min_steplength(const enkf_config_type *config);
void enkf_config_set_min_steplength(enkf_config_type *config,
                                    double min_steplength);

double enkf_config_get_dec_steplength(const enkf_config_type *config);
void enkf_config_set_dec_steplength(enkf_config_type *config,
                                    double dec_steplength);

inversion_type enkf_config_get_inversion(const enkf_config_type *config);
void enkf_config_set_inversion(enkf_config_type *config,
                               inversion_type inversion);

bool enkf_config_get_subspace(const enkf_config_type *config);
void enkf_config_set_subspace(enkf_config_type *config, bool subspace);

bool enkf_config_get_aaprojection(const enkf_config_type *config);
void enkf_config_set_aaprojection(enkf_config_type *config, bool aaprojection);

char *enkf_config_get_logfile(const enkf_config_type *config);
void enkf_config_set_logfile(enkf_config_type *config, const char *logfile);

double enkf_config_calculate_steplength(const enkf_config_type *config,
                                        int iteration_nr);
} // namespace ies
#endif
