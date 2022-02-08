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

#include <string>

#include <ert/util/type_macros.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/bool_vector.hpp>

/*
   These are option flag values which are used by the core ert code to
   query the module of it's needs and capabilities. For instance to to
   determine whether the data should be scaled prior to analysis the
   core code will issue the call:

      if (analysis_module_get_option( module, ANALYSIS_SCALE_DATA))
         obs_data_scale( obs_data , S , E , D , R , dObs );

   It is the responsability of the module to set the various flags.
*/

typedef enum {
    ANALYSIS_NEED_ED = 1,
    ANALYSIS_USE_A =
        4, // The module will read the content of A - but not modify it.
    ANALYSIS_UPDATE_A =
        8, // The update will be based on modifying A directly, and not on an X matrix.
    ANALYSIS_SCALE_DATA = 16,
    ANALYSIS_ITERABLE = 32 // The module can bu used as an iterative smoother.
} analysis_module_flag_enum;

typedef enum {
    ENSEMBLE_SMOOTHER = 1,
    ITERATED_ENSEMBLE_SMOOTHER = 2
} analysis_mode_enum;


class AnalysisModule {
public:
    AnalysisModule(int ens_size, analysis_mode_enum mode);
    AnalysisModule(int ens_size, analysis_mode_enum mode, const std::string& module_name);
    ~AnalysisModule();


    void initX(matrix_type * X, const matrix_type * A, const matrix_type * S, const matrix_type *R, const matrix_type * dObs, const matrix_type *E, const matrix_type * D, rng_type *rng) const;

    updateA(matrix_type *A, const matrix_type * A, const matrix_type *S, const matrix_type *R, const matrix_type * dObs, const matrix_type * E, const matrix_type *D, rng_type * rng);

    init_update(const bool_vector_type *ens_mask,
                const bool_vector_type *obs_mask,
                const matrix_type *S, const matrix_type *R,
                const matrix_type *dObs, const matrix_type *E,
                const matrix_type *D, rng_type *rng) const;

    int ens_size() const;
    analysis_mode_enum mode() const;
    const std::string& name() const;

    bool set(const std::string& var_name, const std::string& value);
    bool has(const std::string& var_name);
    template <typename T>
    T get(const std::string& var) const;

private:
    std::string            user_name;
    analysis_mode_enum     mode;
    ies::data::data_type * module_data;
};


#endif
