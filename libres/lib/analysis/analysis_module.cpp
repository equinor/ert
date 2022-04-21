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

#include <dlfcn.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <memory>
#include <stdexcept>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/ies/ies_data.hpp>

#include <ert/logging.hpp>
#include <ert/python.hpp>
#include <fmt/format.h>

auto logger = ert::get_logger("analysis");

struct analysis_module_struct {
    std::unique_ptr<ies::data::Data> module_data;
    std::unique_ptr<ies::config::Config> module_config;
    char *
        user_name; /* String used to identify this module for the user; not used in
                                                   the linking process. */
    analysis_mode_enum mode;
    std::unordered_set<std::string> keys;
};

analysis_mode_enum
analysis_module_get_mode(const analysis_module_type *module) {
    return module->mode;
}

int analysis_module_ens_size(const analysis_module_type *module) {
    return module->module_data->ens_size();
}

analysis_module_type *analysis_module_alloc_named(int ens_size,
                                                  analysis_mode_enum mode,
                                                  const char *module_name) {
    analysis_module_type *module = new analysis_module_type();

    module->mode = mode;
    module->user_name = util_alloc_string_copy(module_name);
    module->module_config = std::make_unique<ies::config::Config>(
        mode == ITERATED_ENSEMBLE_SMOOTHER);
    module->module_data = std::make_unique<ies::data::Data>(ens_size);
    module->user_name = util_alloc_string_copy(module_name);

    return module;
}

analysis_module_type *analysis_module_alloc(int ens_size,
                                            analysis_mode_enum mode) {
    if (mode == ENSEMBLE_SMOOTHER) {
        analysis_module_type *module = new analysis_module_type();
        module->mode = mode;
        module->user_name = util_alloc_string_copy("STD_ENKF");
        module->module_config = std::make_unique<ies::config::Config>(false);
        module->module_data = std::make_unique<ies::data::Data>(ens_size);
        module->keys = {ies::data::ITER_KEY, ies::config::IES_INVERSION_KEY,
                        ies::config::IES_LOGFILE_KEY,
                        ies::config::IES_DEBUG_KEY,
                        ies::config::ENKF_TRUNCATION_KEY};
        return module;
    } else if (mode == ITERATED_ENSEMBLE_SMOOTHER) {
        analysis_module_type *module = new analysis_module_type();
        module->mode = mode;
        module->user_name = util_alloc_string_copy("IES_ENKF");
        module->module_config = std::make_unique<ies::config::Config>(true);
        module->module_data = std::make_unique<ies::data::Data>(ens_size);
        module->keys = {ies::data::ITER_KEY,
                        ies::config::IES_MAX_STEPLENGTH_KEY,
                        ies::config::IES_MIN_STEPLENGTH_KEY,
                        ies::config::IES_DEC_STEPLENGTH_KEY,
                        ies::config::IES_INVERSION_KEY,
                        ies::config::IES_LOGFILE_KEY,
                        ies::config::IES_DEBUG_KEY,
                        ies::config::ENKF_TRUNCATION_KEY};
        return module;
    } else
        throw std::logic_error("Unhandled enum value");
}

const char *analysis_module_get_name(const analysis_module_type *module) {
    return module->user_name;
}

void analysis_module_free(analysis_module_type *module) {
    free(module->user_name);
    delete module;
}

static bool analysis_module_set_int(analysis_module_type *module,
                                    const char *flag, int value) {
    if (strcmp(flag, ies::config::ENKF_NCOMP_KEY) == 0)
        module->module_config->subspace_dimension(value);

    else if (strcmp(flag, ies::config::ENKF_SUBSPACE_DIMENSION_KEY) == 0)
        module->module_config->subspace_dimension(value);

    else if (strcmp(flag, ies::data::ITER_KEY) == 0)
        module->module_data->iteration_nr(value);

    else if (strcmp(flag, ies::config::IES_INVERSION_KEY) == 0)
        module->module_config->inversion(
            static_cast<ies::config::inversion_type>(value));

    else
        return false;

    return true;
}

int analysis_module_get_int(const analysis_module_type *module,
                            const char *var) {

    if (strcmp(var, ies::config::ENKF_NCOMP_KEY) == 0 ||
        strcmp(var, ies::config::ENKF_SUBSPACE_DIMENSION_KEY) == 0) {
        const auto &truncation = module->module_config->truncation();
        if (std::holds_alternative<int>(truncation))
            return std::get<int>(truncation);
        else
            return -1;
    }

    else if (strcmp(var, ies::data::ITER_KEY) == 0)
        return module->module_data->iteration_nr();

    else if (strcmp(var, ies::config::IES_INVERSION_KEY) == 0)
        return module->module_config->inversion();

    util_exit("%s: Tried to get integer variable:%s from module:%s - "
              "module does not support this variable \n",
              __func__, var, module->user_name);

    return 0;
}

static bool analysis_module_set_double(analysis_module_type *module,
                                       const char *var, double value) {
    bool name_recognized = true;

    if (strcmp(var, ies::config::ENKF_TRUNCATION_KEY) == 0)
        module->module_config->truncation(value);
    else if (strcmp(var, ies::config::IES_MAX_STEPLENGTH_KEY) == 0)
        module->module_config->max_steplength(value);
    else if (strcmp(var, ies::config::IES_MIN_STEPLENGTH_KEY) == 0)
        module->module_config->min_steplength(value);
    else if (strcmp(var, ies::config::IES_DEC_STEPLENGTH_KEY) == 0)
        module->module_config->dec_steplength(value);
    else
        name_recognized = false;

    return name_recognized;
}

static bool analysis_module_set_bool(analysis_module_type *module,
                                     const char *var, bool value) {
    bool name_recognized = true;
    if (strcmp(var, ies::config::IES_DEBUG_KEY) == 0)
        logger->warning("The key {} is ignored", ies::config::IES_DEBUG_KEY);
    else
        name_recognized = false;

    return name_recognized;
}

static bool analysis_module_set_string(analysis_module_type *module,
                                       const char *var, const char *value) {
    bool valid_set = true;
    if (strcmp(var, ies::config::INVERSION_KEY) == 0) {
        if (strcmp(value, ies::config::STRING_INVERSION_SUBSPACE_EXACT_R) == 0)
            module->module_config->inversion(
                ies::config::IES_INVERSION_SUBSPACE_EXACT_R);

        else if (strcmp(value, ies::config::STRING_INVERSION_SUBSPACE_EE_R) ==
                 0)
            module->module_config->inversion(
                ies::config::IES_INVERSION_SUBSPACE_EE_R);

        else if (strcmp(value, ies::config::STRING_INVERSION_SUBSPACE_RE) == 0)
            module->module_config->inversion(
                ies::config::IES_INVERSION_SUBSPACE_RE);

        else if (strcmp(var, ies::config::IES_LOGFILE_KEY) == 0)
            logger->warning("The key {} is ignored",
                            ies::config::IES_LOGFILE_KEY);

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

bool analysis_module_has_var(const analysis_module_type *module,
                             const char *var) {
    return module->keys.count(var);
}

bool analysis_module_get_bool(const analysis_module_type *module,
                              const char *var) {
    if (strcmp(var, ies::config::IES_DEBUG_KEY) == 0)
        return false;

    util_exit("%s: Tried to get bool variable:%s from module:%s - module "
              "does not support this variable \n",
              __func__, var, module->user_name);

    return false;
}

double analysis_module_get_double(const analysis_module_type *module,
                                  const char *var) {

    if (strcmp(var, ies::config::ENKF_TRUNCATION_KEY) == 0) {
        const auto &truncation = module->module_config->truncation();
        if (std::holds_alternative<double>(truncation))
            return std::get<double>(truncation);
        else
            return -1;
    }

    else if (strcmp(var, ies::config::IES_MAX_STEPLENGTH_KEY) == 0)
        return module->module_config->max_steplength();

    else if (strcmp(var, ies::config::IES_MIN_STEPLENGTH_KEY) == 0)
        return module->module_config->min_steplength();

    else if (strcmp(var, ies::config::IES_DEC_STEPLENGTH_KEY) == 0)
        return module->module_config->dec_steplength();

    util_exit("%s: Tried to get double variable:%s from module:%s - module "
              "does not support this variable \n",
              __func__, var, module->user_name);

    return -1;
}

ies::data::Data *
analysis_module_get_module_data(const analysis_module_type *module) {
    return module->module_data.get();
}

ies::config::Config *
analysis_module_get_module_config(const analysis_module_type *module) {
    return module->module_config.get();
}

ies::config::Config *
analysis_module_get_module_config_pybind(py::object module) {
    auto module_ = ert::from_cwrap<analysis_module_type>(module);
    return analysis_module_get_module_config(module_);
}

ies::data::Data *analysis_module_get_module_data_pybind(py::object module) {
    auto module_ = ert::from_cwrap<analysis_module_type>(module);
    return analysis_module_get_module_data(module_);
}

RES_LIB_SUBMODULE("analysis_module", m) {
    m.def("get_module_config", analysis_module_get_module_config_pybind,
          py::return_value_policy::reference_internal);
    m.def("get_module_data", analysis_module_get_module_data_pybind,
          py::return_value_policy::reference_internal);
}
