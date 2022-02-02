/*
   Copyright (C) 2019  Equinor ASA, Norway.

   The file 'ies_config.cpp' is part of ERT - Ensemble based Reservoir Tool.

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
#include <cmath>
#include <stdexcept>
#include <variant>

#include <ert/util/util.hpp>
#include <ert/util/type_macros.hpp>

#include <ert/analysis/std_enkf.hpp>
#include <ert/analysis/analysis_module.hpp>

#include <ert/analysis/ies/ies_config.hpp>

#define DEFAULT_IES_MAX_STEPLENGTH 0.60
#define DEFAULT_IES_MIN_STEPLENGTH 0.30
#define DEFAULT_IES_DEC_STEPLENGTH 2.50
#define MIN_IES_DEC_STEPLENGTH 1.1
#define DEFAULT_IES_INVERSION ies::config::IES_INVERSION_SUBSPACE_EXACT_R
#define DEFAULT_IES_AAPROJECTION false

#define IES_CONFIG_TYPE_ID 196402021

struct ies::config::config_struct {
    UTIL_TYPE_ID_DECLARATION;
    std::variant<double, int> truncation;
    long option_flags;
    double
        ies_max_steplength; // Controlled by config key: DEFAULT_IES_MAX_STEPLENGTH_KEY
    double
        ies_min_steplength; // Controlled by config key: DEFAULT_IES_MIN_STEPLENGTH_KEY
    double
        ies_dec_steplength; // Controlled by config key: DEFAULT_IES_DEC_STEPLENGTH_KEY
    inversion_type
        ies_inversion;     // Controlled by config key: DEFAULT_IES_INVERSION
    bool ies_aaprojection; // Controlled by config key: DEFAULT_IES_AAPROJECTION
};

ies::config::config_type *ies::config::alloc(bool ies_mode) {
    ies::config::config_type *config = new ies::config::config_type();
    UTIL_TYPE_ID_INIT(config, IES_CONFIG_TYPE_ID);
    ies::config::set_truncation(config, DEFAULT_TRUNCATION);
    if (ies_mode)
        ies::config::set_option_flags(
            config, ANALYSIS_NEED_ED + ANALYSIS_UPDATE_A + ANALYSIS_ITERABLE +
                        ANALYSIS_SCALE_DATA);
    else
        ies::config::set_option_flags(config,
                                      ANALYSIS_NEED_ED + ANALYSIS_SCALE_DATA);

    ies::config::set_max_steplength(config, DEFAULT_IES_MAX_STEPLENGTH);
    ies::config::set_min_steplength(config, DEFAULT_IES_MIN_STEPLENGTH);
    ies::config::set_dec_steplength(config, DEFAULT_IES_DEC_STEPLENGTH);
    ies::config::set_inversion(config, DEFAULT_IES_INVERSION);
    ies::config::set_aaprojection(config, DEFAULT_IES_AAPROJECTION);

    return config;
}

/*------------------------------------------------------------------------------------------------*/
/* TRUNCATION -> SUBSPACE_DIMENSION */

const std::variant<double, int> &
ies::config::get_truncation(const config_type *config) {
    return config->truncation;
}

void ies::config::set_truncation(config_type *config, double truncation) {
    config->truncation = truncation;
}

void ies::config::set_subspace_dimension(config_type *config,
                                         int subspace_dimension) {
    config->truncation = subspace_dimension;
}

/*------------------------------------------------------------------------------------------------*/
/* OPTION_FLAGS */

long ies::config::get_option_flags(const ies::config::config_type *config) {
    return config->option_flags;
}

void ies::config::set_option_flags(config_type *config, long flags) {
    config->option_flags = flags;
}

bool ies::config::get_option(const config_type *config,
                             analysis_module_flag_enum option) {
    return ((config->option_flags & option) == option);
}

void ies::config::set_option(config_type *config,
                             analysis_module_flag_enum option) {
    config->option_flags |= option;
}

void ies::config::del_option(config_type *config,
                             analysis_module_flag_enum option) {
    if (ies::config::get_option(config, option))
        config->option_flags -= option;
}

/*------------------------------------------------------------------------------------------------*/
/* IES_MAX_STEPLENGTH */
double ies::config::get_max_steplength(const ies::config::config_type *config) {
    return config->ies_max_steplength;
}
void ies::config::set_max_steplength(ies::config::config_type *config,
                                     double ies_max_steplength) {
    config->ies_max_steplength = ies_max_steplength;
}
/*------------------------------------------------------------------------------------------------*/
/* IES_MIN_STEPLENGTH */
double ies::config::get_min_steplength(const ies::config::config_type *config) {
    return config->ies_min_steplength;
}
void ies::config::set_min_steplength(ies::config::config_type *config,
                                     double ies_min_steplength) {
    config->ies_min_steplength = ies_min_steplength;
}

/*------------------------------------------------------------------------------------------------*/
/* IES_DEC_STEPLENGTH */
double ies::config::get_dec_steplength(const ies::config::config_type *config) {
    return config->ies_dec_steplength;
}
void ies::config::set_dec_steplength(ies::config::config_type *config,
                                     double ies_dec_steplength) {

    // The formula used to calculate step length has a hard assumption that the
    // steplength is reduced for every step - here that is silently enforced
    // with the std::max(1.1, ....).
    config->ies_dec_steplength =
        std::max(ies_dec_steplength, MIN_IES_DEC_STEPLENGTH);
}

/*------------------------------------------------------------------------------------------------*/
/* IES_INVERSION          */
ies::config::inversion_type
ies::config::get_inversion(const ies::config::config_type *config) {
    return config->ies_inversion;
}
void ies::config::set_inversion(ies::config::config_type *config,
                                ies::config::inversion_type ies_inversion) {
    config->ies_inversion = ies_inversion;
}

/*------------------------------------------------------------------------------------------------*/
/* IES_AAPROJECTION         */
bool ies::config::get_aaprojection(const ies::config::config_type *config) {
    return config->ies_aaprojection;
}
void ies::config::set_aaprojection(config_type *config, bool ies_aaprojection) {
    config->ies_aaprojection = ies_aaprojection;
}

/*------------------------------------------------------------------------------------------------*/
/* FREE_CONFIG */
void ies::config::free(ies::config::config_type *config) { delete config; }

double
ies::config::calculate_steplength(const ies::config::config_type *ies_config,
                                  int iteration_nr) {
    double ies_max_step = ies::config::get_max_steplength(ies_config);
    double ies_min_step = ies::config::get_min_steplength(ies_config);
    double ies_decline_step = ies::config::get_dec_steplength(ies_config);

    /*
      This is an implementation of Eq. (49) from the book:

      Geir Evensen, Formulating the history matching problem with consistent error statistics,
      Computational Geosciences (2021) 25:945 –970: https://doi.org/10.1007/s10596-021-10032-7
    */

    double ies_steplength =
        ies_min_step + (ies_max_step - ies_min_step) *
                           pow(2, -(iteration_nr - 1) / (ies_decline_step - 1));

    return ies_steplength;
}
