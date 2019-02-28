/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'callback_arg.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdlib.h>

#include <ert/enkf/callback_arg.hpp>

#define CALLBACK_ARG_TYPE_ID 7814509


void callback_arg_free(callback_arg_type * cb_arg) {
  free( cb_arg );
}



callback_arg_type * callback_arg_alloc(const res_config_type * res_config,
                                       run_arg_type * run_arg,
                                       rng_type * rng)
{
  callback_arg_type * cb = (callback_arg_type *)util_malloc( sizeof * cb );
  UTIL_TYPE_ID_INIT( cb , CALLBACK_ARG_TYPE_ID);
  cb->run_arg = run_arg;
  cb->rng = rng;
  cb->res_config = res_config;
  return cb;
}



UTIL_IS_INSTANCE_FUNCTION( callback_arg, CALLBACK_ARG_TYPE_ID )
UTIL_SAFE_CAST_FUNCTION( callback_arg, CALLBACK_ARG_TYPE_ID )


