/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'analysis_module.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unordered_map>
#include <unordered_set>

#include <stdexcept>

#include <ert/res_util/matrix.hpp>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/ies/ies_data.hpp>

#include <fmt/format.h>
#include <ert/logging.hpp>

auto logger = ert::get_logger("analysis");

#define ANALYSIS_MODULE_TYPE_ID 6610123

struct analysis_module_struct {
    UTIL_TYPE_ID_DECLARATION;
    ies::data::data_type *module_data;
    char *
        user_name; /* String used to identify this module for the user; not used in
                                                   the linking process. */
    analysis_mode_enum mode;
};

analysis_mode_enum
analysis_module_get_mode(const analysis_module_type *module) {
    return module->mode;
}

int analysis_module_ens_size(const analysis_module_type *module) {
    return ies::data::ens_size(module->module_data);
}

analysis_module_type *analysis_module_alloc_named(int ens_size,
                                                  analysis_mode_enum mode,
                                                  const char *module_name) {
    analysis_module_type *module =
        (analysis_module_type *)util_malloc(sizeof *module);
    UTIL_TYPE_ID_INIT(module, ANALYSIS_MODULE_TYPE_ID);

    module->mode = mode;
    module->module_data = NULL;
    module->user_name = util_alloc_string_copy(module_name);
    module->module_data =
        ies::data::alloc(ens_size, mode == ITERATED_ENSEMBLE_SMOOTHER);
    module->user_name = util_alloc_string_copy(module_name);

    return module;
}

analysis_module_type *analysis_module_alloc(int ens_size,
                                            analysis_mode_enum mode) {
    if (mode == ENSEMBLE_SMOOTHER)
        return analysis_module_alloc_named(ens_size, mode, "STD_ENKF");
    else if (mode == ITERATED_ENSEMBLE_SMOOTHER)
        return analysis_module_alloc_named(ens_size, mode, "IES_ENKF");
    else
        throw std::logic_error("Undandled enum value");
}

const char *analysis_module_get_name(const analysis_module_type *module) {
    return module->user_name;
}

static UTIL_SAFE_CAST_FUNCTION(analysis_module, ANALYSIS_MODULE_TYPE_ID)
    UTIL_IS_INSTANCE_FUNCTION(analysis_module, ANALYSIS_MODULE_TYPE_ID)

        void analysis_module_free(analysis_module_type *module) {
    ies::data::free(module->module_data);
    free(module->user_name);
    free(module);
}

void analysis_module_initX(analysis_module_type *module, matrix_type *X,
                           const matrix_type *A, const matrix_type *S,
                           const matrix_type *R, const matrix_type *dObs,
                           const matrix_type *E, const matrix_type *D,
                           rng_type *rng) {
    ies::initX(module->module_data, S, R, E, D, X);
}

void analysis_module_updateA(analysis_module_type *module, matrix_type *A,
                             const matrix_type *S, const matrix_type *R,
                             const matrix_type *dObs, const matrix_type *E,
                             const matrix_type *D, rng_type *rng) {

    ies::updateA(module->module_data, A, S, R, dObs, E, D, rng);
}

void analysis_module_init_update(analysis_module_type *module,
                                 const bool_vector_type *ens_mask,
                                 const bool_vector_type *obs_mask,
                                 const matrix_type *S, const matrix_type *R,
                                 const matrix_type *dObs, const matrix_type *E,
                                 const matrix_type *D, rng_type *rng) {

    /*
    The ensemble and observation masks sent to the init_update() function can be
    misleading? When assembling the S,R,E and D matrices the inactive
    observations & realisatons have been filtered out completely, i.e. the
    ens_mask and obs_mask variables are *not* used to filter out rows and
    columns from the S,R,E and D matrices.

    In the case of multi iteration updates we need to detect the changes in
    active realisatons/observations between iterations, and that is the purpose
    of the ens_mask and obs_mask variables.
  */

    if (bool_vector_count_equal(ens_mask, true) != matrix_get_columns(S))
        throw std::invalid_argument(
            "Internal error - number of columns in S must be equal to number "
            "of *active* realisatons");

    if (bool_vector_count_equal(obs_mask, true) != matrix_get_rows(S))
        throw std::invalid_argument(
            "Internal error - number of rows in S must be equal to number of "
            "*active* observations");

    ies::init_update(module->module_data, ens_mask, obs_mask, S, R, dObs, E, D,
                     rng);
}

static bool analysis_module_set_int(analysis_module_type *module,
                                    const char *flag, int value) {

    auto &ies_config = ies::data::get_config(module->module_data);
    if (strcmp(flag, ies::config::ENKF_NCOMP_KEY) == 0)
        ies_config.subspace_dimension(value);

    else if (strcmp(flag, ies::config::ENKF_SUBSPACE_DIMENSION_KEY) == 0)
        ies_config.subspace_dimension(value);

    else if (strcmp(flag, ies::data::ITER_KEY) == 0)
        ies::data::set_iteration_nr(module->module_data, value);

    else if (strcmp(flag, ies::config::IES_INVERSION_KEY) == 0)
        ies_config.inversion(static_cast<ies::config::inversion_type>(value));

    else
        return false;

    return true;
}

int analysis_module_get_int(const analysis_module_type *module,
                            const char *var) {

    const auto &ies_config = ies::data::get_config(module->module_data);
    if (strcmp(var, ies::config::ENKF_NCOMP_KEY) == 0 ||
        strcmp(var, ies::config::ENKF_SUBSPACE_DIMENSION_KEY) == 0) {
        const auto &truncation = ies_config.truncation();
        if (std::holds_alternative<int>(truncation))
            return std::get<int>(truncation);
        else
            return -1;
    }

    else if (strcmp(var, ies::data::ITER_KEY) == 0)
        return ies::data::get_iteration_nr(module->module_data);

    else if (strcmp(var, ies::config::IES_INVERSION_KEY) == 0)
        return ies_config.inversion();

    util_exit("%s: Tried to get integer variable:%s from module:%s - "
              "module does not support this variable \n",
              __func__, var, module->user_name);

    return 0;
}

static bool analysis_module_set_double(analysis_module_type *module,
                                       const char *var, double value) {
    auto &ies_config = ies::data::get_config(module->module_data);
    bool name_recognized = true;

    if (strcmp(var, ies::config::ENKF_TRUNCATION_KEY) == 0)
        ies_config.truncation(value);
    else if (strcmp(var, ies::config::IES_MAX_STEPLENGTH_KEY) == 0)
        ies_config.max_steplength(value);
    else if (strcmp(var, ies::config::IES_MIN_STEPLENGTH_KEY) == 0)
        ies_config.min_steplength(value);
    else if (strcmp(var, ies::config::IES_DEC_STEPLENGTH_KEY) == 0)
        ies_config.dec_steplength(value);
    else
        name_recognized = false;

    return name_recognized;
}

static bool analysis_module_set_bool(analysis_module_type *module,
                                     const char *var, bool value) {
    auto &ies_config = ies::data::get_config(module->module_data);
    bool name_recognized = true;

    if (strcmp(var, ies::config::ANALYSIS_SCALE_DATA_KEY) == 0) {
        if (value)
            ies_config.set_option(ANALYSIS_SCALE_DATA);
        else
            ies_config.del_option(ANALYSIS_SCALE_DATA);
    } else if (strcmp(var, ies::config::IES_AAPROJECTION_KEY) == 0)
        ies_config.aaprojection(value);
    else if (strcmp(var, ies::config::IES_DEBUG_KEY) == 0)
        logger->warning("The key {} is ignored", ies::config::IES_DEBUG_KEY);
    else
        name_recognized = false;

    return name_recognized;
}

static bool analysis_module_set_string(analysis_module_type *module,
                                       const char *var, const char *value) {
    auto &ies_config = ies::data::get_config(module->module_data);
    bool valid_set = true;
    if (strcmp(var, ies::config::INVERSION_KEY) == 0) {
        if (strcmp(value, ies::config::STRING_INVERSION_SUBSPACE_EXACT_R) == 0)
            ies_config.inversion(ies::config::IES_INVERSION_SUBSPACE_EXACT_R);

        else if (strcmp(value, ies::config::STRING_INVERSION_SUBSPACE_EE_R) ==
                 0)
            ies_config.inversion(ies::config::IES_INVERSION_SUBSPACE_EE_R);

        else if (strcmp(value, ies::config::STRING_INVERSION_SUBSPACE_RE) == 0)
            ies_config.inversion(ies::config::IES_INVERSION_SUBSPACE_RE);

        else
            valid_set = false;
    } else
        valid_set = false;

    return valid_set;
}

/*
   The input value typically comes from the configuration system and
   is in terms of a string, irrespective of the fundamental type of
   the underlying parameter. The algorithm for setting the parameter
   tries datatypes as follows: integer - double - string.

   For the numeric datatypes the algorithm is two step:

     1. Try the conversion string -> numeric.
     2. Try calling the analysis_module_set_xxx() function.

   Observe that this implies that the same variable name can NOT be
   used for different variable types.
*/

bool analysis_module_set_var(analysis_module_type *module, const char *var_name,
                             const char *string_value) {
    logger->info(
        fmt::format("Setting '{}' to value: '{}'", var_name, string_value));
    bool set_ok = false;
    {
        int int_value;

        if (util_sscanf_int(string_value, &int_value))
            set_ok = analysis_module_set_int(module, var_name, int_value);

        if (set_ok)
            return true;
    }

    {
        double double_value;
        if (util_sscanf_double(string_value, &double_value))
            set_ok = analysis_module_set_double(module, var_name, double_value);

        if (set_ok)
            return true;
    }

    {
        bool bool_value;
        if (util_sscanf_bool(string_value, &bool_value))
            set_ok = analysis_module_set_bool(module, var_name, bool_value);

        if (strcmp(var_name, "USE_EE") == 0)
            logger->warning("The USE_EE/USE_GE settings have been removed - "
                            "use the INVERSION setting instead");

        if (strcmp(var_name, "USE_GE") == 0)
            logger->warning("The USE_EE/USE_GE settings have been removed - "
                            "use the INVERSION setting instead");

        if (set_ok)
            return true;
    }

    set_ok = analysis_module_set_string(module, var_name, string_value);
    if (!set_ok)
        fprintf(stderr,
                "** Warning: failed to set %s=%s for analysis module:%s\n",
                var_name, string_value, module->user_name);

    return set_ok;
}

bool analysis_module_check_option(const analysis_module_type *module,
                                  analysis_module_flag_enum option) {
    auto &ies_config = ies::data::get_config(module->module_data);
    return ies_config.get_option(option);
}

bool analysis_module_has_var(const analysis_module_type *module,
                             const char *var) {

    static const std::unordered_set<std::string> keys = {
        ies::data::ITER_KEY,
        ies::config::IES_MAX_STEPLENGTH_KEY,
        ies::config::IES_MIN_STEPLENGTH_KEY,
        ies::config::IES_DEC_STEPLENGTH_KEY,
        ies::config::IES_INVERSION_KEY,
        ies::config::IES_LOGFILE_KEY,
        ies::config::IES_DEBUG_KEY,
        ies::config::IES_AAPROJECTION_KEY,
        ies::config::ENKF_TRUNCATION_KEY,
        ies::config::ENKF_SUBSPACE_DIMENSION_KEY,
        ies::config::ANALYSIS_SCALE_DATA_KEY,
        ies::config::ENKF_NCOMP_KEY};

    return (keys.count(var) == 1);
}

bool analysis_module_get_bool(const analysis_module_type *module,
                              const char *var) {
    auto &ies_config = ies::data::get_config(module->module_data);
    if (strcmp(var, ies::config::ANALYSIS_SCALE_DATA_KEY) == 0)
        return ies_config.get_option(ANALYSIS_SCALE_DATA);

    else if (strcmp(var, ies::config::IES_AAPROJECTION_KEY) == 0)
        return ies_config.aaprojection();

    else if (strcmp(var, ies::config::IES_DEBUG_KEY) == 0)
        return false;

    util_exit("%s: Tried to get bool variable:%s from module:%s - module "
              "does not support this variable \n",
              __func__, var, module->user_name);

    return false;
}

double analysis_module_get_double(const analysis_module_type *module,
                                  const char *var) {

    auto &ies_config = ies::data::get_config(module->module_data);
    if (strcmp(var, ies::config::ENKF_TRUNCATION_KEY) == 0) {
        const auto &truncation = ies_config.truncation();
        if (std::holds_alternative<double>(truncation))
            return std::get<double>(truncation);
        else
            return -1;
    }

    else if (strcmp(var, ies::config::IES_MAX_STEPLENGTH_KEY) == 0)
        return ies_config.max_steplength();

    else if (strcmp(var, ies::config::IES_MIN_STEPLENGTH_KEY) == 0)
        return ies_config.min_steplength();

    else if (strcmp(var, ies::config::IES_DEC_STEPLENGTH_KEY) == 0)
        return ies_config.dec_steplength();

    util_exit("%s: Tried to get double variable:%s from module:%s - module "
              "does not support this variable \n",
              __func__, var, module->user_name);

    return -1;
}
