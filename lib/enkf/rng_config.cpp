/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'rng_config.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/mzran.h>
#include <ert/util/util.h>
#include <ert/util/test_util.h>

#include <ert/config/config_parser.hpp>
#include <ert/config/config_schema_item.hpp>

#include <ert/res_util/res_log.hpp>

#include <ert/enkf/rng_config.hpp>
#include <ert/enkf/rng_manager.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/model_config.hpp>


struct rng_config_struct {
  rng_alg_type      type;
  char            * random_seed;
};


static void rng_config_set_random_seed(rng_config_type * rng_config, const char * random_seed) {
  rng_config->random_seed = util_realloc_string_copy(rng_config->random_seed, random_seed);
}

static char * rng_config_alloc_formatted_random_seed(const rng_config_type * rng_config) {
  unsigned int * fseed = (unsigned int*) malloc(RNG_STATE_SIZE * sizeof(unsigned int*));

  int seed_len = strlen(rng_config->random_seed);
  int seed_pos = 0;
  for (int i = 0; i < RNG_STATE_SIZE; ++i) {
    fseed[i] = 0;
    for (int k = 0; k < RNG_STATE_DIGITS; ++k) {
      fseed[i] *= 10;
      fseed[i] += rng_config->random_seed[seed_pos] - '0';
      seed_pos = (seed_pos+1) % seed_len;
    }
  }

  return (char *) fseed;
}

const char * rng_config_get_random_seed(const rng_config_type * rng_config) {
  return rng_config->random_seed;
}

void rng_config_set_type( rng_config_type * rng_config , rng_alg_type type) {
  rng_config->type = type;
}

rng_alg_type rng_config_get_type(const rng_config_type * rng_config ) {
  return rng_config->type;
}

static rng_config_type * rng_config_alloc_default(void) {
  rng_config_type * rng_config = (rng_config_type *)util_malloc( sizeof * rng_config);

  rng_config_set_type( rng_config , MZRAN );  /* Only type ... */
  rng_config->random_seed = NULL;

  return rng_config;
}

rng_config_type * rng_config_alloc_load_user_config(const char * user_config_file) {
  config_parser_type * config_parser = config_alloc();
  config_content_type * config_content = NULL;
  if(user_config_file)
    config_content = model_config_alloc_content(user_config_file, config_parser);

  rng_config_type * rng_config = rng_config_alloc(config_content);

  config_content_free(config_content);
  config_free(config_parser);

  return rng_config;
}

rng_config_type * rng_config_alloc(const config_content_type * config_content) {
  rng_config_type * rng_config = rng_config_alloc_default();

  if(config_content)
    rng_config_init(rng_config, config_content);

  return rng_config;
}
rng_config_type * rng_config_alloc_full(const char * random_seed) {
  rng_config_type * rng_config = rng_config_alloc_default();
  rng_config->random_seed = util_realloc_string_copy(rng_config->random_seed, random_seed);

  return rng_config;
}


void rng_config_free( rng_config_type * rng) {
  free( rng->random_seed );
  free( rng );
}


rng_manager_type * rng_config_alloc_rng_manager( const rng_config_type * rng_config ) {
  rng_manager_type * rng_manager;

  if (rng_config->random_seed) {
    char * formatted_seed = rng_config_alloc_formatted_random_seed(rng_config);
    rng_manager = rng_manager_alloc(formatted_seed);
  } else {
    rng_manager = rng_manager_alloc_random( );
  }

  rng_manager_log_state(rng_manager);

  return rng_manager;
}


/*****************************************************************/

void rng_config_add_config_items( config_parser_type * parser ) {
  config_add_key_value(parser, RANDOM_SEED_KEY, false, CONFIG_STRING);
}


void rng_config_init(rng_config_type * rng_config, const config_content_type * config_content) {
  if(config_content_has_item(config_content, RANDOM_SEED_KEY)) {
    const char * random_seed = config_content_get_value(config_content, RANDOM_SEED_KEY);
    rng_config_set_random_seed(rng_config, random_seed);
    res_log_fcritical("Using RANDOM_SEED: %s", random_seed);
  }
}
