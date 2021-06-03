/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'rng_manager.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_RNG_MANAGER_H
#define ERT_RNG_MANAGER_H

#include <ert/util/type_macros.h>
#include <ert/util/rng.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The number of unsigned int's necessary to represent the state of the rng
 * algorithm. Since the current algorithm used is mzran, this value is set to
 * 4.
 */
#define RNG_STATE_SIZE 4

/**
 * The number of digits used to print each unigned int representing the rng
 * state.
 */
#define RNG_STATE_DIGITS 10


typedef struct rng_manager_struct rng_manager_type;

rng_manager_type * rng_manager_alloc(const char * random_seed);
rng_manager_type * rng_manager_alloc_load( const char * seed_file );
rng_manager_type * rng_manager_alloc_default( );
rng_manager_type * rng_manager_alloc_random( );

rng_type         * rng_manager_alloc_rng(rng_manager_type * rng_manager);
rng_type         * rng_manager_iget(rng_manager_type * rng_manager, int index);
void               rng_manager_free( rng_manager_type * rng_manager );
void               rng_manager_save_state(const rng_manager_type * rng_manager, const char * seed_file);
void               rng_manager_log_state(const rng_manager_type * rng_manager);


UTIL_IS_INSTANCE_HEADER( rng_manager );

#ifdef __cplusplus
}
#endif
#endif
