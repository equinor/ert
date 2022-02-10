/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'analysis_module.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_ANALYSIS_MODULE_H
#define ERT_ANALYSIS_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/res_util/matrix.hpp>
#include <ert/util/bool_vector.hpp>

enum analysis_module_flag_enum : int {
    ANALYSIS_USE_A =
        4, // The module will read the content of A - but not modify it.
    ANALYSIS_UPDATE_A =
        8, // The update will be based on modifying A directly, and not on an X matrix.
    ANALYSIS_ITERABLE = 32 // The module can bu used as an iterative smoother.
};

typedef enum {
    ENSEMBLE_SMOOTHER = 1,
    ITERATED_ENSEMBLE_SMOOTHER = 2
} analysis_mode_enum;

typedef struct analysis_module_struct analysis_module_type;

analysis_module_type *analysis_module_alloc(int ens_size,
                                            analysis_mode_enum mode);
analysis_module_type *analysis_module_alloc_named(int ens_size,
                                                  analysis_mode_enum mode,
                                                  const char *module_name);

void analysis_module_free(analysis_module_type *module);

bool analysis_module_set_var(analysis_module_type *module, const char *var_name,
                             const char *string_value);
analysis_mode_enum analysis_module_get_mode(const analysis_module_type *module);
const char *analysis_module_get_name(const analysis_module_type *module);
bool analysis_module_check_option(const analysis_module_type *module,
                                  analysis_module_flag_enum option);

bool analysis_module_has_var(const analysis_module_type *module,
                             const char *var);
double analysis_module_get_double(const analysis_module_type *module,
                                  const char *var);
int analysis_module_get_int(const analysis_module_type *module,
                            const char *var);
bool analysis_module_get_bool(const analysis_module_type *module,
                              const char *var);
void *analysis_module_get_ptr(const analysis_module_type *module,
                              const char *var);
int analysis_module_ens_size(const analysis_module_type *module);

extern "C++" {
#include <ert/analysis/ies/ies_data.hpp>
ies::data::Data *
analysis_module_get_module_data(const analysis_module_type *module);
}

#ifdef __cplusplus
}
#endif

#endif
