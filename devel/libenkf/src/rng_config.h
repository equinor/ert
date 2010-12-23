#ifndef __RNG_CONFIG_H__
#define __RNG_CONFIG_H__

#ifdef __cplusplus 
extern "C" {
#endif

#include <config.h>
#include <rng.h>

typedef struct rng_config_struct rng_config_type; 

void              rng_config_fprintf_config( rng_config_type * rng_config , FILE * stream );
void              rng_config_init( rng_config_type * rng_config , config_type * config );
void              rng_config_set_type( rng_config_type * rng_config , rng_alg_type type);
rng_alg_type      rng_config_get_type(const rng_config_type * rng_config );
const char      * rng_config_get_seed_load_file( const rng_config_type * rng_config );
void              rng_config_set_seed_load_file( rng_config_type * rng_config , const char * seed_load_file);
const char      * rng_config_get_seed_store_file( const rng_config_type * rng_config );
void              rng_config_set_seed_store_file( rng_config_type * rng_config , const char * seed_store_file);
rng_config_type * rng_config_alloc( );
void              rng_config_free( rng_config_type * rng);
void              rng_config_add_config_items( config_type * config );

#ifdef __cplusplus 
}
#endif
#endif
