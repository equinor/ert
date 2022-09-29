/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_state_map.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <vector>

#include <stdlib.h>

#include <ert/util/bool_vector.h>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

#include <ert/enkf/state_map.hpp>

void get_test() {
    StateMap state_map(101);
    test_assert_int_equal(STATE_UNDEFINED, state_map.get(0));
    test_assert_int_equal(STATE_UNDEFINED, state_map.get(100));
}

void set_test() {
    StateMap state_map(101);
    state_map.set(0, STATE_INITIALIZED);
    test_assert_int_equal(STATE_INITIALIZED, state_map.get(0));

    state_map.set(100, STATE_INITIALIZED);
    test_assert_int_equal(STATE_INITIALIZED, state_map.get(100));

    test_assert_int_equal(STATE_UNDEFINED, state_map.get(50));
    test_assert_int_equal(101, state_map.size());
}

void load_empty_test() {
    StateMap state_map("File/does/not/exists");
    test_assert_int_equal(0, state_map.size());
}

void test_equal() {
    StateMap state_map1(151);
    StateMap state_map2(151);

    test_assert_true(state_map1 == state_map2);
    for (int i = 0; i < 25; i++) {
        state_map1.set(i, STATE_INITIALIZED);
        state_map2.set(i, STATE_INITIALIZED);
    }
    test_assert_true(state_map1 == state_map2);

    state_map2.set(15, STATE_HAS_DATA);
    test_assert_false(state_map1 == state_map2);
    state_map2.set(15, STATE_LOAD_FAILURE);
    state_map2.set(15, STATE_INITIALIZED);
    test_assert_true(state_map1 == state_map2);

    state_map2.set(150, STATE_INITIALIZED);
    test_assert_false(state_map1 == state_map2);
}

void test_io() {
    ecl::util::TestArea ta("state_map_io");
    {
        StateMap state_map(101);
        state_map.set(0, STATE_INITIALIZED);
        state_map.set(100, STATE_INITIALIZED);
        state_map.write("map");

        StateMap copy1("map");
        test_assert_true(state_map == copy1);

        StateMap copy2(101);
        test_assert_true(copy2.read("map"));
        test_assert_true(state_map == copy2);

        copy2.set(67, STATE_INITIALIZED);
        test_assert_false(state_map == copy2);

        copy2.read("map");
        test_assert_true(state_map == copy2);
    }
}

void test_update_matching() {
    StateMap state_map(11);

    state_map.set(10, STATE_INITIALIZED);
    state_map.set(3, STATE_PARENT_FAILURE);
    test_assert_int_equal(STATE_UNDEFINED, state_map.get(5));
    test_assert_int_equal(STATE_INITIALIZED, state_map.get(10));

    state_map.update_matching(5, STATE_UNDEFINED | STATE_LOAD_FAILURE,
                              STATE_INITIALIZED);
    state_map.update_matching(10, STATE_UNDEFINED | STATE_LOAD_FAILURE,
                              STATE_INITIALIZED);
    state_map.update_matching(3, STATE_UNDEFINED | STATE_LOAD_FAILURE,
                              STATE_INITIALIZED);

    test_assert_int_equal(STATE_INITIALIZED, state_map.get(5));
    test_assert_int_equal(STATE_INITIALIZED, state_map.get(10));
    test_assert_int_equal(STATE_PARENT_FAILURE, state_map.get(3));

    state_map.update_matching(10, STATE_UNDEFINED, STATE_INITIALIZED);
    test_assert_int_equal(STATE_INITIALIZED, state_map.get(10));
}

void test_select_matching() {
    StateMap state_map(51);

    state_map.set(10, STATE_INITIALIZED);
    state_map.set(10, STATE_HAS_DATA);
    state_map.set(20, STATE_INITIALIZED);
    auto mask = state_map.select_matching(STATE_HAS_DATA | STATE_INITIALIZED);
    test_assert_int_equal(mask.size(), 51);
    test_assert_true(mask[10]);
    test_assert_true(mask[20]);

    mask = state_map.select_matching(STATE_HAS_DATA);

    for (size_t i; i < mask.size(); i++) {
        if (i == 10)
            test_assert_true(mask[i]);
        else {
            test_assert_false(mask[0]);
        }
    }

    state_map.set(50, STATE_INITIALIZED);
    mask = state_map.select_matching(STATE_HAS_DATA | STATE_INITIALIZED);
    test_assert_int_equal(mask.size(), 51);
}

// Probably means that the target should be explicitly set to
// undefined before workflows which automatically change case.
void test_transitions() {

    test_assert_false(
        StateMap::is_legal_transition(STATE_UNDEFINED, STATE_UNDEFINED));
    test_assert_true(
        StateMap::is_legal_transition(STATE_UNDEFINED, STATE_INITIALIZED));
    test_assert_false(
        StateMap::is_legal_transition(STATE_UNDEFINED, STATE_HAS_DATA));
    test_assert_false(
        StateMap::is_legal_transition(STATE_UNDEFINED, STATE_LOAD_FAILURE));
    test_assert_true(
        StateMap::is_legal_transition(STATE_UNDEFINED, STATE_PARENT_FAILURE));

    test_assert_false(
        StateMap::is_legal_transition(STATE_INITIALIZED, STATE_UNDEFINED));
    test_assert_true(
        StateMap::is_legal_transition(STATE_INITIALIZED, STATE_INITIALIZED));
    test_assert_true(
        StateMap::is_legal_transition(STATE_INITIALIZED, STATE_HAS_DATA));
    test_assert_true(
        StateMap::is_legal_transition(STATE_INITIALIZED, STATE_LOAD_FAILURE));
    test_assert_true(StateMap::is_legal_transition(
        STATE_INITIALIZED,
        STATE_PARENT_FAILURE)); // Should maybe false - if the commenta baove is taken into account.

    test_assert_false(
        StateMap::is_legal_transition(STATE_HAS_DATA, STATE_UNDEFINED));
    test_assert_true(
        StateMap::is_legal_transition(STATE_HAS_DATA, STATE_INITIALIZED));
    test_assert_true(
        StateMap::is_legal_transition(STATE_HAS_DATA, STATE_HAS_DATA));
    test_assert_true(
        StateMap::is_legal_transition(STATE_HAS_DATA, STATE_LOAD_FAILURE));
    test_assert_true(
        StateMap::is_legal_transition(STATE_HAS_DATA,
                                      STATE_PARENT_FAILURE)); // Rerun

    test_assert_false(
        StateMap::is_legal_transition(STATE_LOAD_FAILURE, STATE_UNDEFINED));
    test_assert_true(
        StateMap::is_legal_transition(STATE_LOAD_FAILURE, STATE_INITIALIZED));
    test_assert_true(
        StateMap::is_legal_transition(STATE_LOAD_FAILURE, STATE_HAS_DATA));
    test_assert_true(
        StateMap::is_legal_transition(STATE_LOAD_FAILURE, STATE_LOAD_FAILURE));
    test_assert_false(StateMap::is_legal_transition(STATE_LOAD_FAILURE,
                                                    STATE_PARENT_FAILURE));

    test_assert_false(
        StateMap::is_legal_transition(STATE_PARENT_FAILURE, STATE_UNDEFINED));
    test_assert_true(
        StateMap::is_legal_transition(STATE_PARENT_FAILURE, STATE_INITIALIZED));
    test_assert_false(
        StateMap::is_legal_transition(STATE_PARENT_FAILURE, STATE_HAS_DATA));
    test_assert_false(StateMap::is_legal_transition(STATE_PARENT_FAILURE,
                                                    STATE_LOAD_FAILURE));
    test_assert_true(StateMap::is_legal_transition(STATE_PARENT_FAILURE,
                                                   STATE_PARENT_FAILURE));
}

int main(int argc, char **argv) {
    get_test();
    set_test();
    load_empty_test();
    test_equal();
    test_io();
    test_select_matching();
    test_transitions();
    exit(0);
}
