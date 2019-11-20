/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'rng_config.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_RNG_CONFIG_H
#define ERT_RNG_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/config/config_parser.hpp>
#include <ert/config/config_content.hpp>

#include <ert/enkf/rng_manager.hpp>

typedef struct rng_config_struct rng_config_type;

  void               rng_config_init(rng_config_type * rng_config, const config_content_type * config);
  void               rng_config_set_type( rng_config_type * rng_config , rng_alg_type type);
  rng_alg_type       rng_config_get_type(const rng_config_type * rng_config );
  const char       * rng_config_get_seed_load_file( const rng_config_type * rng_config );
  const char       * rng_config_get_random_seed(const rng_config_type * rng_config);
  void               rng_config_set_seed_load_file( rng_config_type * rng_config , const char * seed_load_file);
  const char       * rng_config_get_seed_store_file( const rng_config_type * rng_config );
  void               rng_config_set_seed_store_file( rng_config_type * rng_config , const char * seed_store_file);
  rng_config_type  * rng_config_alloc_load_user_config(const char * user_config_file);
  rng_config_type  * rng_config_alloc(const config_content_type * config_content);
  rng_config_type  * rng_config_alloc_full(const char * random_seed, const char * store_seed, const char * load_seed);
  void               rng_config_free( rng_config_type * rng);
  void               rng_config_add_config_items( config_parser_type * config );
  rng_manager_type * rng_config_alloc_rng_manager( const rng_config_type * rng_config );

#ifdef __cplusplus
}
#endif
#endif
