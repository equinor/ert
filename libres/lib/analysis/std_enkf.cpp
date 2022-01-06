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
#define INVALID_SUBSPACE_DIMENSION -1
#define INVALID_TRUNCATION -1
#define DEFAULT_SUBSPACE_DIMENSION INVALID_SUBSPACE_DIMENSION
#define DEFAULT_USE_EE false
#define DEFAULT_USE_GE false
#define DEFAULT_ANALYSIS_SCALE_DATA true

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
    double truncation; // Controlled by config key: ENKF_TRUNCATION_KEY
    int subspace_dimension; // Controlled by config key: ENKF_NCOMP_KEY (-1: use Truncation instead)
    long option_flags;
    bool
        __use_EE; // Deprecated flag - see comment above function: update_inversion_enum
    bool
        __use_GE; // Deprecated flag - see comment above function: update_inversion_enum
    ies::config::inversion_type inversion_type;
    bool analysis_scale_data;
};

static UTIL_SAFE_CAST_FUNCTION_CONST(std_enkf_data, STD_ENKF_TYPE_ID)

    /*
  This is a macro which will expand to generate a function:

     std_enkf_data_type * std_enkf_data_safe_cast( void * arg ) {}

  which is used for runtime type checking of all the functions which
  accept a void pointer as first argument.
*/
    static UTIL_SAFE_CAST_FUNCTION(std_enkf_data, STD_ENKF_TYPE_ID)

        double std_enkf_get_truncation(std_enkf_data_type *data) {
    return data->truncation;
}

/*
  The std_enkf module originally used two boolean flags use_EE and use_GE to
  determine how the inversion should be performed. In order to harmonize with
  the ies_enkf module the code is updated to rather use an enum from the
  ies_enkf code. Since the use_GE and use_EE flags can be set as a string option
  from user code it is retained with some ugly code to ensure that both setting
  based on the USE_EE/USE_GHE settings and the new INVERSION setting should give
  the same result. The bool flags __use_EE and __use_GE are extremely deprecated
  and should be removed.
*/

static void update_inversion_enum(std_enkf_data_type *std_enkf_data) {
    if (std_enkf_data->__use_EE) {
        if (std_enkf_data->__use_GE)
            std_enkf_data->inversion_type =
                ies::config::IES_INVERSION_SUBSPACE_RE;
        else
            std_enkf_data->inversion_type =
                ies::config::IES_INVERSION_SUBSPACE_EE_R;
    } else
        std_enkf_data->inversion_type =
            ies::config::IES_INVERSION_SUBSPACE_EXACT_R;
}

static void update_inversion_flags(std_enkf_data_type *std_enkf_data) {
    switch (std_enkf_data->inversion_type) {
    case ies::config::IES_INVERSION_SUBSPACE_EXACT_R:
        std_enkf_data->__use_EE = false;
        return;

    case ies::config::IES_INVERSION_SUBSPACE_RE:
        std_enkf_data->__use_EE = true;
        std_enkf_data->__use_GE = true;
        return;

    case ies::config::IES_INVERSION_SUBSPACE_EE_R:
        std_enkf_data->__use_EE = true;
        std_enkf_data->__use_GE = false;
        return;

    default:
        return;
    }
}

int std_enkf_get_subspace_dimension(std_enkf_data_type *data) {
    return data->subspace_dimension;
}

ies::config::inversion_type
std_enkf_data_get_inversion(const std_enkf_data_type *data) {
    return data->inversion_type;
}

void std_enkf_set_truncation(std_enkf_data_type *data, double truncation) {
    data->truncation = truncation;
    if (truncation > 0.0)
        data->subspace_dimension = INVALID_SUBSPACE_DIMENSION;
}

void std_enkf_set_subspace_dimension(std_enkf_data_type *data,
                                     int subspace_dimension) {
    data->subspace_dimension = subspace_dimension;
    if (subspace_dimension > 0)
        data->truncation = INVALID_TRUNCATION;
}

void *std_enkf_data_alloc() {
    std_enkf_data_type *data = (std_enkf_data_type *)util_malloc(sizeof *data);
    UTIL_TYPE_ID_INIT(data, STD_ENKF_TYPE_ID);

    std_enkf_set_truncation(data, DEFAULT_ENKF_TRUNCATION_);
    std_enkf_set_subspace_dimension(data, DEFAULT_SUBSPACE_DIMENSION);
    data->option_flags = ANALYSIS_NEED_ED;
    data->__use_EE = DEFAULT_USE_EE;
    data->__use_GE = DEFAULT_USE_GE;
    data->analysis_scale_data = DEFAULT_ANALYSIS_SCALE_DATA;
    data->inversion_type = ies::config::IES_INVERSION_SUBSPACE_EXACT_R;
    return data;
}

void std_enkf_data_free(void *data) { free(data); }

static void std_enkf_initX__(matrix_type *X, const matrix_type *S0,
                             const matrix_type *R, const matrix_type *E,
                             const matrix_type *D, double truncation, int ncomp,
                             bool bootstrap,
                             ies::config::inversion_type inversion_type) {

    matrix_type *S = matrix_alloc_copy(S0);
    int nrobs = matrix_get_rows(S);
    int ens_size = matrix_get_columns(S);
    int nrmin = std::min(ens_size, nrobs);

    matrix_type *W = matrix_alloc(nrobs, nrmin);
    std::vector<double> eig(nrmin);

    matrix_subtract_row_mean(S); /* Shift away the mean */

    if (inversion_type == ies::config::IES_INVERSION_SUBSPACE_RE)
        enkf_linalg_lowrankE(S, E, W, eig.data(), truncation, ncomp);
    else if (inversion_type == ies::config::IES_INVERSION_SUBSPACE_EE_R) {
        matrix_type *Et = matrix_alloc_transpose(E);
        matrix_type *Cee = matrix_alloc_matmul(E, Et);
        matrix_scale(Cee, 1.0 / (ens_size - 1));

        enkf_linalg_lowrankCinv(S, Cee, W, eig.data(), truncation, ncomp);

        matrix_free(Et);
        matrix_free(Cee);
    } else if (inversion_type == ies::config::IES_INVERSION_SUBSPACE_EXACT_R)
        enkf_linalg_lowrankCinv(S, R, W, eig.data(), truncation, ncomp);

    enkf_linalg_init_stdX(X, S, D, W, eig.data(), bootstrap);

    matrix_free(W);
    matrix_free(S);
    enkf_linalg_checkX(X, bootstrap);
}

void std_enkf_initX(void *module_data, matrix_type *X, const matrix_type *A,
                    const matrix_type *S, const matrix_type *R,
                    const matrix_type *dObs, const matrix_type *E,
                    const matrix_type *D, rng_type *rng) {

    std_enkf_data_type *data = std_enkf_data_safe_cast(module_data);
    {
        int ncomp = data->subspace_dimension;
        double truncation = data->truncation;

        std_enkf_initX__(X, S, R, E, D, truncation, ncomp, false,
                         data->inversion_type);
    }
}

bool std_enkf_set_double(void *arg, const char *var_name, double value) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    {
        bool name_recognized = true;

        if (strcmp(var_name, ENKF_TRUNCATION_KEY_) == 0)
            std_enkf_set_truncation(module_data, value);
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
            std_enkf_set_subspace_dimension(module_data, value);
        else
            name_recognized = false;

        return name_recognized;
    }
}

bool std_enkf_set_bool(void *arg, const char *var_name, bool value) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    {
        bool name_recognized = true;

        if (strcmp(var_name, USE_EE_KEY_) == 0)
            module_data->__use_EE = value;
        else if (strcmp(var_name, USE_GE_KEY_) == 0)
            module_data->__use_GE = value;
        else if (strcmp(var_name, ANALYSIS_SCALE_DATA_KEY_) == 0)
            module_data->analysis_scale_data = value;
        else
            name_recognized = false;

        update_inversion_enum(module_data);
        return name_recognized;
    }
}

bool std_enkf_set_string(void *arg, const char *var_name, const char *value) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    {
        bool valid_set = true;
        if (strcmp(var_name, INVERSION_KEY) == 0) {
            if (strcmp(value, STRING_INVERSION_SUBSPACE_EXACT_R) == 0)
                module_data->inversion_type =
                    ies::config::IES_INVERSION_SUBSPACE_EXACT_R;

            else if (strcmp(value, STRING_INVERSION_SUBSPACE_EE_R) == 0)
                module_data->inversion_type =
                    ies::config::IES_INVERSION_SUBSPACE_EE_R;

            else if (strcmp(value, STRING_INVERSION_SUBSPACE_RE) == 0)
                module_data->inversion_type =
                    ies::config::IES_INVERSION_SUBSPACE_RE;

            else
                valid_set = false;
        } else
            valid_set = false;

        update_inversion_flags(module_data);
        return valid_set;
    }
}

long std_enkf_get_options(void *arg, long flag) {
    std_enkf_data_type *module_data = std_enkf_data_safe_cast(arg);
    int scale_option =
        (module_data->analysis_scale_data) ? ANALYSIS_SCALE_DATA : 0;
    return module_data->option_flags + scale_option;
}

bool std_enkf_has_var(const void *arg, const char *var_name) {
    {
        if (strcmp(var_name, ENKF_NCOMP_KEY_) == 0)
            return true;
        else if (strcmp(var_name, ENKF_TRUNCATION_KEY_) == 0)
            return true;
        else if (strcmp(var_name, USE_EE_KEY_) == 0)
            return true;
        else if (strcmp(var_name, USE_GE_KEY_) == 0)
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
        if (strcmp(var_name, ENKF_TRUNCATION_KEY_) == 0)
            return module_data->truncation;
        else
            return -1;
    }
}

int std_enkf_get_int(const void *arg, const char *var_name) {
    const std_enkf_data_type *module_data = std_enkf_data_safe_cast_const(arg);
    {
        if (strcmp(var_name, ENKF_NCOMP_KEY_) == 0)
            return module_data->subspace_dimension;
        else
            return -1;
    }
}

bool std_enkf_get_bool(const void *arg, const char *var_name) {
    const std_enkf_data_type *module_data = std_enkf_data_safe_cast_const(arg);
    {
        if (strcmp(var_name, USE_EE_KEY_) == 0)
            return module_data->__use_EE;
        else if (strcmp(var_name, USE_GE_KEY_) == 0)
            return module_data->__use_GE;
        else if (strcmp(var_name, ANALYSIS_SCALE_DATA_KEY_) == 0)
            return module_data->analysis_scale_data;
        else
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
