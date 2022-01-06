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

#ifndef IES_CONFIG_H
#define IES_CONFIG_H

#include <stdbool.h>

namespace ies {

typedef enum {
    IES_INVERSION_EXACT = 0,
    IES_INVERSION_SUBSPACE_EXACT_R = 1,
    IES_INVERSION_SUBSPACE_EE_R = 2,
    IES_INVERSION_SUBSPACE_RE = 3
} inversion_type;

typedef struct config_struct config_type;

config_type *config_alloc();
void config_free(config_type *config);

int config_get_subspace_dimension(const config_type *config);
void config_set_subspace_dimension(config_type *config, int subspace_dimension);

double config_get_truncation(const config_type *config);
void config_set_truncation(config_type *config, double truncation);

void config_set_option_flags(config_type *config, long flags);
long config_get_option_flags(const config_type *config);

double config_get_max_steplength(const config_type *config);
void config_set_max_steplength(config_type *config, double max_steplength);

double config_get_min_steplength(const config_type *config);
void config_set_min_steplength(config_type *config, double min_steplength);

double config_get_dec_steplength(const config_type *config);
void config_set_dec_steplength(config_type *config, double dec_steplength);

inversion_type config_get_inversion(const config_type *config);
void config_set_inversion(config_type *config, inversion_type inversion);

bool config_get_subspace(const config_type *config);
void config_set_subspace(config_type *config, bool subspace);

bool config_get_aaprojection(const config_type *config);
void config_set_aaprojection(config_type *config, bool aaprojection);

char *config_get_logfile(const config_type *config);
void config_set_logfile(config_type *config, const char *logfile);

double config_calculate_steplength(const config_type *config, int iteration_nr);
} // namespace ies
#endif
