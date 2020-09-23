/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_util.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_ENKF_UTIL_H
#define ERT_ENKF_UTIL_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <ert/util/buffer.h>
#include <ert/util/rng.h>

#include <ert/ecl/ecl_util.h>
#include <ert/ecl/ecl_type.h>

#include <ert/enkf/enkf_types.hpp>

void    enkf_util_truncate(void *  , int  , ecl_data_type , void *  , void *);
double  enkf_util_rand_normal(double , double , rng_type * rng);
void    enkf_util_assert_buffer_type(buffer_type * buffer, ert_impl_type target_type);

#ifdef __cplusplus
}
#endif
#endif
