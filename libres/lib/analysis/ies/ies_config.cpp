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

#include <ert/analysis/analysis_module.hpp>

#include <ert/analysis/ies/ies_config.hpp>

#define DEFAULT_IES_MAX_STEPLENGTH 0.60
#define DEFAULT_IES_MIN_STEPLENGTH 0.30
#define DEFAULT_IES_DEC_STEPLENGTH 2.50
#define MIN_IES_DEC_STEPLENGTH 1.1
#define DEFAULT_IES_INVERSION ies::config::IES_INVERSION_SUBSPACE_EXACT_R
#define DEFAULT_IES_AAPROJECTION false

ies::config::Config::Config(bool ies_mode)
    : m_truncation(DEFAULT_TRUNCATION), m_ies_inversion(DEFAULT_IES_INVERSION),
      m_ies_aaprojection(DEFAULT_IES_AAPROJECTION),
      m_ies_max_steplength(DEFAULT_IES_MAX_STEPLENGTH),
      m_ies_min_steplength(DEFAULT_IES_MIN_STEPLENGTH),
      m_ies_dec_steplength(DEFAULT_IES_DEC_STEPLENGTH) {
    if (ies_mode)
        this->m_option_flags = ANALYSIS_NEED_ED + ANALYSIS_UPDATE_A +
                               ANALYSIS_SCALE_DATA + ANALYSIS_ITERABLE;
    else
        this->m_option_flags = ANALYSIS_NEED_ED + ANALYSIS_SCALE_DATA;
}

/*------------------------------------------------------------------------------------------------*/
/* TRUNCATION -> SUBSPACE_DIMENSION */

const std::variant<double, int> &ies::config::Config::truncation() const {
    return this->m_truncation;
}

void ies::config::Config::truncation(double truncation) {
    this->m_truncation = truncation;
}

void ies::config::Config::subspace_dimension(int subspace_dimension) {
    this->m_truncation = subspace_dimension;
}

/*------------------------------------------------------------------------------------------------*/
/* OPTION_FLAGS */

long ies::config::Config::get_option_flags() const {
    return this->m_option_flags;
}

void ies::config::Config::set_option_flags(long flags) {
    this->m_option_flags = flags;
}

bool ies::config::Config::get_option(analysis_module_flag_enum option) const {
    return ((this->m_option_flags & option) == option);
}

void ies::config::Config::set_option(analysis_module_flag_enum option) {
    this->m_option_flags |= option;
}

void ies::config::Config::del_option(analysis_module_flag_enum option) {
    if (this->get_option(option))
        this->m_option_flags -= option;
}

double ies::config::Config::max_steplength() const {
    return this->m_ies_max_steplength;
}

void ies::config::Config::max_steplength(double max_step) {
    this->m_ies_max_steplength = max_step;
}

double ies::config::Config::min_steplength() const {
    return this->m_ies_min_steplength;
}

void ies::config::Config::min_steplength(double min_step) {
    this->m_ies_min_steplength = min_step;
}

double ies::config::Config::dec_steplength() const {
    return this->m_ies_dec_steplength;
}

void ies::config::Config::dec_steplength(double dec_step) {
    this->m_ies_dec_steplength = std::max(dec_step, MIN_IES_DEC_STEPLENGTH);
}

/*------------------------------------------------------------------------------------------------*/
/* IES_INVERSION          */
ies::config::inversion_type ies::config::Config::inversion() const {
    return this->m_ies_inversion;
}
void ies::config::Config::inversion(ies::config::inversion_type it) {
    this->m_ies_inversion = it;
}

/*------------------------------------------------------------------------------------------------*/
/* IES_AAPROJECTION         */
bool ies::config::Config::aaprojection() const {
    return this->m_ies_aaprojection;
}

void ies::config::Config::aaprojection(bool ies_aaprojection) {
    this->m_ies_aaprojection = ies_aaprojection;
}

double ies::config::Config::steplength(int iteration_nr) const {
    double ies_max_step = this->m_ies_max_steplength;
    double ies_min_step = this->m_ies_min_steplength;
    double ies_decline_step = this->m_ies_dec_steplength;

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
