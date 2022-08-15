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
#include <ert/python.hpp>
#include <stdexcept>
#include <variant>

#include <ert/analysis/ies/ies_config.hpp>

#define DEFAULT_IES_MAX_STEPLENGTH 0.60
#define DEFAULT_IES_MIN_STEPLENGTH 0.30
#define DEFAULT_IES_DEC_STEPLENGTH 2.50
#define MIN_IES_DEC_STEPLENGTH 1.1
#define DEFAULT_IES_INVERSION ies::IES_INVERSION_EXACT

ies::Config::Config(bool ies_mode)
    : m_truncation(DEFAULT_TRUNCATION), inversion(DEFAULT_IES_INVERSION),
      iterable(ies_mode), max_steplength(DEFAULT_IES_MAX_STEPLENGTH),
      min_steplength(DEFAULT_IES_MIN_STEPLENGTH),
      m_dec_steplength(DEFAULT_IES_DEC_STEPLENGTH) {}

/*------------------------------------------------------------------------------------------------*/
/* TRUNCATION -> SUBSPACE_DIMENSION */

const std::variant<double, int> &ies::Config::get_truncation() const {
    return this->m_truncation;
}

void ies::Config::set_truncation(double truncation) {
    this->m_truncation = truncation;
}

void ies::Config::subspace_dimension(int subspace_dimension) {
    this->m_truncation = subspace_dimension;
}

double ies::Config::get_dec_steplength() const {
    return this->m_dec_steplength;
}

void ies::Config::set_dec_steplength(double dec_step) {
    this->m_dec_steplength = std::max(dec_step, MIN_IES_DEC_STEPLENGTH);
}

double ies::Config::get_steplength(int iteration_nr) const {
    double ies_max_step = this->max_steplength;
    double ies_min_step = this->min_steplength;
    double ies_decline_step = this->m_dec_steplength;

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

ERT_CLIB_SUBMODULE("ies", m) {
    using namespace py::literals;
    py::class_<ies::Config, std::shared_ptr<ies::Config>>(m, "Config")
        .def(py::init<bool>())
        .def("get_steplength", &ies::Config::get_steplength)
        .def("get_truncation", &ies::Config::get_truncation)
        .def_readwrite("iterable", &ies::Config::iterable)
        .def_readwrite("inversion", &ies::Config::inversion);

    py::enum_<ies::inversion_type>(m, "inversion_type")
        .value("EXACT", ies::inversion_type::IES_INVERSION_EXACT)
        .value("EE_R", ies::inversion_type::IES_INVERSION_SUBSPACE_EE_R)
        .value("EXACT_R", ies::inversion_type::IES_INVERSION_SUBSPACE_EXACT_R)
        .value("SUBSPACE_RE", ies::inversion_type::IES_INVERSION_SUBSPACE_RE)
        .export_values();
}
