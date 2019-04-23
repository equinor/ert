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
  char            * seed_load_file;    /* NULL: Do not store the seed. */
  char            * seed_store_file;   /* NULL: Do not load a seed from file. */
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

const char * rng_config_get_seed_load_file( const rng_config_type * rng_config ) {
  return rng_config->seed_load_file;
}

void rng_config_set_seed_load_file( rng_config_type * rng_config , const char * seed_load_file) {
  rng_config->seed_load_file = util_realloc_string_copy( rng_config->seed_load_file , seed_load_file);
}

const char * rng_config_get_seed_store_file( const rng_config_type * rng_config ) {
  return rng_config->seed_store_file;
}

void rng_config_set_seed_store_file( rng_config_type * rng_config , const char * seed_store_file) {
  rng_config->seed_store_file = util_realloc_string_copy( rng_config->seed_store_file , seed_store_file);
}


static rng_config_type * rng_config_alloc_default(void) {
  rng_config_type * rng_config = (rng_config_type *)util_malloc( sizeof * rng_config);

  rng_config_set_type( rng_config , MZRAN );  /* Only type ... */
  rng_config->random_seed = NULL;
  rng_config->seed_store_file = NULL;
  rng_config->seed_load_file = NULL;

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
rng_config_type * rng_config_alloc_full(const char * random_seed, const char * store_seed, const char * load_seed) {
  rng_config_type * rng_config = rng_config_alloc_default();

  rng_config->random_seed = util_realloc_string_copy(rng_config->random_seed, random_seed);
  rng_config->seed_store_file = util_realloc_string_copy(rng_config->seed_store_file, store_seed);
  rng_config->seed_load_file = util_realloc_string_copy(rng_config->seed_load_file, load_seed);

  return rng_config;
}


void rng_config_free( rng_config_type * rng) {
  free( rng->seed_load_file );
  free( rng->seed_store_file );
  free( rng->random_seed );
  free( rng );
}


rng_manager_type * rng_config_alloc_rng_manager( const rng_config_type * rng_config ) {
  const char * seed_store = rng_config_get_seed_store_file( rng_config );
  const char * seed_load  = rng_config_get_seed_load_file( rng_config );
  rng_manager_type * rng_manager;

  if (rng_config->random_seed) {
    char * formatted_seed = rng_config_alloc_formatted_random_seed(rng_config);
    rng_manager = rng_manager_alloc(formatted_seed);
  }
  else if (seed_load && util_file_exists( seed_load )) {
    rng_manager = rng_manager_alloc_load( seed_load );
  }
  else {
    rng_manager = rng_manager_alloc_random( );
  }

  rng_manager_log_state(rng_manager);
  if (seed_store)
    rng_manager_save_state( rng_manager , seed_store );

  return rng_manager;
}


/*****************************************************************/

void rng_config_add_config_items( config_parser_type * parser ) {
  config_add_key_value(parser, STORE_SEED_KEY, false, CONFIG_PATH);
  config_install_message(
          parser,
          STORE_SEED_KEY,
          "WARNING: STORE_SEED is deprecated - for reproducibility, fetch logged RANDOM_SEED instead");

  config_add_key_value(parser, LOAD_SEED_KEY, false, CONFIG_PATH);
  config_install_message(
          parser,
          LOAD_SEED_KEY,
          "WARNING: LOAD_SEED is deprecated - use RANDOM_SEED instead");

  config_add_key_value(parser, RANDOM_SEED_KEY, false, CONFIG_STRING);
}


void rng_config_init(rng_config_type * rng_config, const config_content_type * config_content) {
  if(config_content_has_item(config_content, RANDOM_SEED_KEY)) {
    const char * random_seed = config_content_get_value(config_content, RANDOM_SEED_KEY);
    rng_config_set_random_seed(rng_config, random_seed);
  }

  if(config_content_has_item(config_content, STORE_SEED_KEY)) {
    if(rng_config->random_seed)
      res_log_warning("Cannot have both RANDOM_SEED and STORE_SEED "
                      "keywords. STORE_SEED will be ignored.");
    else
      rng_config_set_seed_store_file(rng_config,
                                     config_content_iget(config_content,
                                                         STORE_SEED_KEY, 0, 0));
  }

  if(config_content_has_item(config_content, LOAD_SEED_KEY)) {
    if(rng_config->random_seed)
      res_log_warning("Cannot have both RANDOM_SEED and LOAD_SEED "
                      "keywords. LOAD_SEED will be ignored.");
    else
      rng_config_set_seed_load_file(rng_config,
                                    config_content_iget(config_content,
                                                        LOAD_SEED_KEY, 0 ,0));
  }
}


void rng_config_fprintf_config( rng_config_type * rng_config , FILE * stream ) {
  if (rng_config->seed_load_file != NULL) {
    fprintf( stream , CONFIG_KEY_FORMAT      , LOAD_SEED_KEY );
    fprintf( stream , CONFIG_ENDVALUE_FORMAT , rng_config->seed_load_file);
  }

  if (rng_config->seed_store_file != NULL) {
    fprintf( stream , CONFIG_KEY_FORMAT      , STORE_SEED_KEY );
    fprintf( stream , CONFIG_ENDVALUE_FORMAT , rng_config->seed_store_file);
  }
}
