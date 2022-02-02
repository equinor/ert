/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'std_enkf.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <algorithm>
#include <vector>

#include <stdlib.h>
#include <string.h>

#include <ert/util/util.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/res_util/matrix_blas.hpp>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/analysis_table.hpp>
#include <ert/analysis/enkf_linalg.hpp>
#include <ert/analysis/std_enkf.hpp>
#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/ies/ies.hpp>

/*
  A random 'magic' integer id which is used for run-time type checking
  of the input data.
*/
#define STD_ENKF_TYPE_ID 261123

/*
  Observe that only one of the settings subspace_dimension and
  truncation can be valid at a time; otherwise the svd routine will
  fail. This implies that the set_truncation() and
  set_subspace_dimension() routines will set one variable, AND
  INVALIDATE THE OTHER. For most situations this will be OK, but if
  you have repeated calls to both of these functions the end result
  might be a surprise.
*/

/*
  The configuration data used by the std_enkf module is contained in a
  std_enkf_data_struct instance. The data type used for the std_enkf
  module is quite simple; with only a few scalar variables, but there
  are essentially no limits to what you can pack into such a datatype.

  All the functions in the module have a void pointer as the first
  argument, this will immediately be casted to a std_enkf_data_type
  instance, to get some type safety the UTIL_TYPE_ID system should be
  used (see documentation in util.h)

  The data structure holding the data for your analysis module should
  be created and initialized by a constructor, which should be
  registered with the '.alloc' element of the analysis table; in the
  same manner the desctruction of this data should be handled by a
  destructor or free() function registered with the .freef field of
  the analysis table.
*/

struct std_enkf_data_struct {
    UTIL_TYPE_ID_DECLARATION;
    ies::config::config_type *ies_config;
};

static UTIL_SAFE_CAST_FUNCTION_CONST(std_enkf_data, STD_ENKF_TYPE_ID)

    /*
  This is a macro which will expand to generate a function:

     std_enkf_data_type * std_enkf_data_safe_cast( void * arg ) {}

  which is used for runtime type checking of all the functions which
  accept a void pointer as first argument.
*/
    static UTIL_SAFE_CAST_FUNCTION(std_enkf_data, STD_ENKF_TYPE_ID)

        const std::variant<double, int> &std_enkf_get_truncation(
            std_enkf_data_type *data) {
    return ies::config::get_truncation(data->ies_config);
}


ies::config::inversion_type
std_enkf_data_get_inversion(const std_enkf_data_type *data) {
    return ies::config::get_inversion(data->ies_config);
}

void std_enkf_set_truncation(std_enkf_data_type *data, double truncation) {
    ies::config::set_truncation(data->ies_config, truncation);
}

void std_enkf_set_subspace_dimension(std_enkf_data_type *data,
                                     int subspace_dimension) {
    ies::config::set_subspace_dimension(data->ies_config, subspace_dimension);
}

const ies::config::config_type *
std_enkf_data_get_config(const std_enkf_data_type *data) {
    return data->ies_config;
}

void *std_enkf_data_alloc() {
    std_enkf_data_type *data = (std_enkf_data_type *)util_malloc(sizeof *data);
    UTIL_TYPE_ID_INIT(data, STD_ENKF_TYPE_ID);
    data->ies_config = ies::config::alloc();
    ies::config::set_truncation(data->ies_config, DEFAULT_ENKF_TRUNCATION_);
    ies::config::set_inversion(data->ies_config,
                               ies::config::IES_INVERSION_SUBSPACE_EXACT_R);
    ies::config::set_option_flags(data->ies_config,
                                  ANALYSIS_NEED_ED + ANALYSIS_SCALE_DATA);

    return data;
}

void std_enkf_data_free(void *data) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(data);
    ies::config::free(module_data->ies_config);
    free(data);
}

void std_enkf_initX(void *module_data, matrix_type *X, const matrix_type *A,
                    const matrix_type *S, const matrix_type *R,
                    const matrix_type *dObs, const matrix_type *E,
                    const matrix_type *D, rng_type *rng) {

    std_enkf_data_type *data = std_enkf_data_safe_cast(module_data);
    ies::initX(data->ies_config, S, R, E, D, X);
}

bool std_enkf_set_double(void *arg, const char *var_name, double value) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    {
        bool name_recognized = true;

        if (strcmp(var_name, ENKF_TRUNCATION_KEY_) == 0)
            ies::config::set_truncation(module_data->ies_config, value);
        else
            name_recognized = false;

        return name_recognized;
    }
}

bool std_enkf_set_int(void *arg, const char *var_name, int value) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    {
        bool name_recognized = true;

        if (strcmp(var_name, ENKF_NCOMP_KEY_) == 0)
            ies::config::set_subspace_dimension(module_data->ies_config, value);
        else
            name_recognized = false;

        return name_recognized;
    }
}

bool std_enkf_set_bool(void *arg, const char *var_name, bool value) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    {
        bool name_recognized = true;

        if (strcmp(var_name, ANALYSIS_SCALE_DATA_KEY_) == 0) {
            if (value)
                ies::config::set_option(module_data->ies_config,
                                        ANALYSIS_SCALE_DATA);
            else
                ies::config::del_option(module_data->ies_config,
                                        ANALYSIS_SCALE_DATA);
        } else
            name_recognized = false;

        return name_recognized;
    }
}

bool std_enkf_set_string(void *arg, const char *var_name, const char *value) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    {
        bool valid_set = true;
        if (strcmp(var_name, INVERSION_KEY) == 0) {
            if (strcmp(value, STRING_INVERSION_SUBSPACE_EXACT_R) == 0)
                ies::config::set_inversion(
                    module_data->ies_config,
                    ies::config::IES_INVERSION_SUBSPACE_EXACT_R);

            else if (strcmp(value, STRING_INVERSION_SUBSPACE_EE_R) == 0)
                ies::config::set_inversion(
                    module_data->ies_config,
                    ies::config::IES_INVERSION_SUBSPACE_EE_R);

            else if (strcmp(value, STRING_INVERSION_SUBSPACE_RE) == 0)
                ies::config::set_inversion(
                    module_data->ies_config,
                    ies::config::IES_INVERSION_SUBSPACE_RE);

            else
                valid_set = false;
        } else
            valid_set = false;

        return valid_set;
    }
}

long std_enkf_get_options(void *arg, long flag) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    return ies::config::get_option_flags(module_data->ies_config);
}

bool std_enkf_has_var(const void *arg, const char *var_name) {
    {
        if (strcmp(var_name, ENKF_NCOMP_KEY_) == 0)
            return true;
        else if (strcmp(var_name, ENKF_TRUNCATION_KEY_) == 0)
            return true;
        else if (strcmp(var_name, ANALYSIS_SCALE_DATA_KEY_) == 0)
            return true;
        else
            return false;
    }
}

double std_enkf_get_double(const void *arg, const char *var_name) {
    const std_enkf_data_type *module_data = std_enkf_data_safe_cast_const(arg);
    {
        if (strcmp(var_name, ENKF_TRUNCATION_KEY_) == 0) {
            const auto &truncation =
                ies::config::get_truncation(module_data->ies_config);
            if (std::holds_alternative<double>(truncation))
                return std::get<double>(truncation);
            else
                return -1;
        } else
            return -1;
    }
}

int std_enkf_get_int(const void *arg, const char *var_name) {
    const std_enkf_data_type *module_data = std_enkf_data_safe_cast_const(arg);
    {
        if (strcmp(var_name, ENKF_NCOMP_KEY_) == 0) {
            const auto &truncation =
                ies::config::get_truncation(module_data->ies_config);
            if (std::holds_alternative<int>(truncation))
                return std::get<int>(truncation);
            else
                return -1;
        } else
            return -1;
    }
}

bool std_enkf_get_bool(const void *arg, const char *var_name) {
    const std_enkf_data_type *module_data = std_enkf_data_safe_cast_const(arg);
    {
        if (strcmp(var_name, ANALYSIS_SCALE_DATA_KEY_) == 0) {
            auto flags = ies::config::get_option_flags(module_data->ies_config);
            return (flags & ANALYSIS_SCALE_DATA);
        } else
            return false;
    }
}

analysis_table_type STD_ENKF = {
    .name = "STD_ENKF",
    .updateA = NULL,
    .initX = std_enkf_initX,
    .init_update = NULL,
    .complete_update = NULL,

    .freef = std_enkf_data_free,
    .alloc = std_enkf_data_alloc,

    .set_int = std_enkf_set_int,
    .set_double = std_enkf_set_double,
    .set_bool = std_enkf_set_bool,
    .set_string = std_enkf_set_string,
    .get_options = std_enkf_get_options,

    .has_var = std_enkf_has_var,
    .get_int = std_enkf_get_int,
    .get_double = std_enkf_get_double,
    .get_bool = std_enkf_get_bool,
    .get_ptr = NULL,
};
