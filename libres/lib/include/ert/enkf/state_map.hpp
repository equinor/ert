/*
   Copyright (C) 2013  Equinor ASA, Norway.
   The file 'state_map.c' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_STATE_MAP_H
#define ERT_STATE_MAP_H

#include <vector>
#include <ert/util/type_macros.h>
#include <ert/util/bool_vector.h>

#include <ert/enkf/enkf_types.hpp>

typedef struct state_map_struct state_map_type;

extern "C" state_map_type *state_map_alloc();
state_map_type *state_map_fread_alloc(const char *filename);
state_map_type *state_map_fread_alloc_readonly(const char *filename);
state_map_type *state_map_alloc_copy(const state_map_type *map);
extern "C" bool state_map_is_readonly(const state_map_type *state_map);
extern "C" void state_map_free(state_map_type *map);
extern "C" int state_map_get_size(const state_map_type *map);
extern "C" realisation_state_enum state_map_iget(const state_map_type *map,
                                                 int index);
void state_map_update_undefined(state_map_type *map, int index,
                                realisation_state_enum new_state);
void state_map_update_matching(state_map_type *map, int index, int state_mask,
                               realisation_state_enum new_state);
extern "C" void state_map_iset(state_map_type *map, int index,
                               realisation_state_enum state);
extern "C" bool state_map_equal(const state_map_type *map1,
                                const state_map_type *map2);
extern "C" void state_map_fwrite(const state_map_type *map,
                                 const char *filename);
extern "C" bool state_map_fread(state_map_type *map, const char *filename);
std::vector<bool> state_map_select_matching(const state_map_type *map,
                                            int select_mask, bool select);
void state_map_set_from_inverted_mask(state_map_type *map,
                                      const std::vector<bool> &mask,
                                      realisation_state_enum state);
void state_map_set_from_mask(state_map_type *map, const std::vector<bool> &mask,
                             realisation_state_enum state);
int state_map_count_matching(const state_map_type *state_map, int mask);
extern "C" bool state_map_legal_transition(realisation_state_enum state1,
                                           realisation_state_enum state2);


#endif
