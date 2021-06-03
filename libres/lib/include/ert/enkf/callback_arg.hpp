/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'callback_arg.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef CALLBACK_ARG_H
#define CALLBACK_ARG_H

#include <ert/util/type_macros.h>
#include <ert/util/rng.hpp>

#include <ert/enkf/run_arg.hpp>
#include <ert/enkf/res_config.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct callback_arg_struct callback_arg_type;

struct callback_arg_struct {
  UTIL_TYPE_ID_DECLARATION;
  const res_config_type * res_config;
  run_arg_type    * run_arg;
  rng_type        * rng;
};


callback_arg_type * callback_arg_alloc(const res_config_type * res_config,
                                       run_arg_type * run_arg,
                                       rng_type * rng);

UTIL_IS_INSTANCE_HEADER( callback_arg );
UTIL_SAFE_CAST_HEADER( callback_arg );

#ifdef __cplusplus
}
#endif
#endif
