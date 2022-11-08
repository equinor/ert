#include <vector>

#include <stdlib.h>

#include <ert/util/bool_vector.h>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

#include <ert/enkf/state_map.hpp>

void load_empty_test() {
    StateMap state_map("File/does/not/exists");
    test_assert_int_equal(0, state_map.size());
}

void test_io() {
    ecl::util::TestArea ta("state_map_io");
    {
        StateMap state_map(101);
        state_map[0] = State::initialized;
        state_map[100] = State::initialized;
        state_map.write("map");

        StateMap copy1("map");
        test_assert_true(state_map == copy1);

        StateMap copy2(101);
        test_assert_true(copy2.read("map"));
        test_assert_true(state_map == copy2);

        copy2[67] = State::initialized;
        test_assert_false(state_map == copy2);

        copy2.read("map");
        test_assert_true(state_map == copy2);
    }
}

int main(int argc, char **argv) {
    load_empty_test();
    test_io();
    exit(0);
}
